"""PhotoLab GPU 加速（可选）：OpenCL 融合内核。

设计：
- 通过 PyOpenCL 自动探测 OpenCL 设备，只使用 GPU（NVIDIA GeForce / AMD / Intel
  均走同一套 OpenCL 路径，与厂商无关）；找不到 GPU 时视为不可用，
  core 自动回退到 CPU（NumPy 分块并行）。
- 只加速"纯逐像素"阶段：白平衡 -> 基础 -> 曲线 -> 颜色分级 -> 黑白，
  整条链融合成单个内核，一次上传、一次下载、无中间缓冲。
  含效果（卷积类）/蒙版/取色器时仍走 CPU（见 core._gpu_eligible），
  避免设备间来回搬运抵消收益。
- GPU 与 CPU 的浮点实现存在微小舍入差异，属正常（视觉不可辨）。
- 环境变量 PHOTOLAB_NO_GPU=1 可强制关闭。
"""
from __future__ import annotations

import threading
import traceback

import numpy as np

_lock = threading.Lock()
_state = None          # None=未初始化, False=不可用, dict=就绪
_fail_logged = False

_KERNEL = r"""
float sstep(float e0, float e1, float x) {
    float t = (x - e0) * (1.0f / (e1 - e0));
    t = clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// 基础影调（白平衡 -> 曝光 -> 对比度 -> 高光/阴影/白色/黑色 -> 饱和度/自然饱和度）
// 抽成一个设备函数：整链内核与「只做基础」内核共用同一份实现，
// 两条 GPU 路径才不会各自漂移。公式逐行对应 core.apply_basic。
inline float3 basic_step(float3 v,
    const float gr, const float gg, const float gb,
    const float exposure_mult,
    const float contrast_a,
    const float hi_k,
    const float sh_k1, const float sh_k2,
    const float wh_k1, const float wh_k2,
    const float bl_k,
    const float sat_mult,
    const float vib_a)
{
    const float3 LUMW = (float3)(0.2126f, 0.7152f, 0.0722f);
    v *= (float3)(gr, gg, gb);
    v *= exposure_mult;

    if (contrast_a != 0.0f) {
        float a = contrast_a;
        float3 s = v * v * 3.0f - v * v * v * 2.0f;   // smoothstep(v)
        if (a > 0.0f) {
            float3 s2 = s * s * 3.0f - s * s * s * 2.0f;
            v = v * (1.0f - a) + s2 * a;
        } else {
            float c = 2.0f * a;
            float3 t = v * (1.0f - v) * (v - 0.5f);
            v = v + c * t;
        }
    }
    if (hi_k != 0.0f)
        v *= 1.0f + hi_k * sstep(0.35f, 0.95f, dot(v, LUMW));
    if (sh_k1 != 0.0f || sh_k2 != 0.0f) {
        float mw = 1.0f - sstep(0.0f, 0.45f, dot(v, LUMW));
        v *= 1.0f + sh_k1 * mw;
        v += sh_k2 * mw;
    }
    if (wh_k1 != 0.0f || wh_k2 != 0.0f) {
        float mw = sstep(0.55f, 1.0f, dot(v, LUMW));
        v *= 1.0f + wh_k1 * mw;
        v += wh_k2 * mw;
    }
    if (bl_k != 0.0f)
        v += bl_k * (1.0f - sstep(0.0f, 0.40f, dot(v, LUMW)));
    if (sat_mult != 1.0f) {
        float lum = dot(v, LUMW);
        v = lum + (v - lum) * sat_mult;
    }
    if (vib_a != 0.0f) {
        float lum = dot(v, LUMW);
        float mx = max(v.x, max(v.y, v.z));
        float mn = min(v.x, min(v.y, v.z));
        float sat = (mx - mn) / (mx + mn + 1e-5f);
        float wv = (vib_a > 0.0f ? 1.0f - sat : sat) * vib_a + 1.0f;
        v = lum + (v - lum) * wv;
    }
    return v;
}

// 只做基础影调，输入输出都是 float32 线性光（不裁切、不编码）。
// 有效果算子的渲染路径用它替换 CPU 的 apply_basic，效果/蒙版仍留在 CPU。
__kernel void basic_frame(
    __global const float *src,
    __global float *dst,
    const int w,
    const int h,
    const float gr, const float gg, const float gb,
    const float exposure_mult,
    const float contrast_a,
    const float hi_k,
    const float sh_k1, const float sh_k2,
    const float wh_k1, const float wh_k2,
    const float bl_k,
    const float sat_mult,
    const float vib_a)
{
    int i = get_global_id(0);
    if (i >= w * h)
        return;
    float3 v = basic_step(vload3(i, src), gr, gg, gb, exposure_mult,
                          contrast_a, hi_k, sh_k1, sh_k2, wh_k1, wh_k2,
                          bl_k, sat_mult, vib_a);
    vstore3(v, i, dst);
}

// 与 core.py 一致的整条逐像素链：
// 基础影调 -> 曲线 LUT -> 颜色分级 -> 黑白 -> sRGB 编码
__kernel void process_frame(
    __global const float *src,
    __global const float *luts,      // 3*lut_n，主通道已复合，R/G/B 依次排列
    __global const float *woff,      // nwheels * 3 色轮偏移向量
    __global const float *wexp,      // nwheels 蒙版指数
    __global const int   *wkind,     // nwheels 范围类别 0阴影 1中间调 2高光 3全局
    __global const uchar *srgb_lut,  // 65536 项 sRGB 编码表（与 CPU 完全同一张表）
    const int nwheels,
    const int lut_n,
    const int use_curves,            // 曲线为恒等时与 CPU 一样跳过该步
    const int w,
    const int h,
    const float gr, const float gg, const float gb,   // 白平衡增益
    const float exposure_mult,
    const float contrast_a,
    const float hi_k,
    const float sh_k1, const float sh_k2,
    const float wh_k1, const float wh_k2,
    const float bl_k,
    const float sat_mult,
    const float vib_a,
    const float bw_k,
    __global uchar *dst)
{
    const float3 LUMW = (float3)(0.2126f, 0.7152f, 0.0722f);
    int i = get_global_id(0);
    if (i >= w * h)
        return;
    float3 v = basic_step(vload3(i, src), gr, gg, gb, exposure_mult,
                          contrast_a, hi_k, sh_k1, sh_k2, wh_k1, wh_k2,
                          bl_k, sat_mult, vib_a);

    // 曲线：LUT 查表（与 CPU 一致：截断取整后夹取；恒等曲线时与 CPU 一样跳过）
    if (use_curves != 0) {
        int ir = clamp(convert_int_sat(v.x * (lut_n - 1)), 0, lut_n - 1);
        int ig = clamp(convert_int_sat(v.y * (lut_n - 1)), 0, lut_n - 1);
        int ib = clamp(convert_int_sat(v.z * (lut_n - 1)), 0, lut_n - 1);
        v = (float3)(luts[ir], luts[lut_n + ig], luts[2 * lut_n + ib]);
    }

    if (nwheels > 0) {
        float lum = dot(v, LUMW);
        for (int k = 0; k < nwheels; ++k) {
            float m;
            int kd = wkind[k];
            if (kd == 0)
                m = 1.0f - sstep(0.0f, 0.5f, lum);
            else if (kd == 1) {
                m = 1.0f - fabs(2.0f * lum - 1.0f) * 1.6f;
                m = clamp(m, 0.0f, 1.0f);
            } else if (kd == 2)
                m = sstep(0.5f, 1.0f, lum);
            else
                m = 1.0f;
            m = pow(m, wexp[k]);
            v += m * vload3(k, woff);
        }
    }

    if (bw_k != 0.0f) {
        float lum = dot(v, LUMW);
        v = v * (1.0f - bw_k) + lum * bw_k;
    }

    // 裁剪 -> sRGB 编码 -> uint8（16 位量化查表，与 CPU 的 linear_to_rgb8 同一张表，
    // 因此 GPU 预览与 CPU 保存按位一致，不受 float32 pow 的舍入差异影响）
    v = clamp(v, 0.0f, 1.0f);
    // rint 的「四舍六入五成双」由 rint() 完成（convert_int_sat 只做截断），与 CPU 的 np.rint 对齐
    int ir = clamp(convert_int_sat(rint(v.x * 65535.0f)), 0, 65535);
    int ig = clamp(convert_int_sat(rint(v.y * 65535.0f)), 0, 65535);
    int ib = clamp(convert_int_sat(rint(v.z * 65535.0f)), 0, 65535);
    uchar3 o = (uchar3)(srgb_lut[ir], srgb_lut[ig], srgb_lut[ib]);
    vstore3(o, i, dst);
}
"""


def _init():
    """惰性初始化 OpenCL；返回状态 dict 或 False。"""
    global _state
    with _lock:
        if _state is not None:
            return _state
        try:
            import pyopencl as cl
            gpus = []
            plat_name = ""
            for plat in cl.get_platforms():
                for dev in plat.get_devices(cl.device_type.GPU):
                    gpus.append(dev)
                    plat_name = plat_name or plat.name
            if not gpus:
                raise RuntimeError("未找到 OpenCL GPU 设备")
            dev = max(gpus, key=lambda d: d.max_compute_units)
            ctx = cl.Context([dev])
            prog = cl.Program(ctx, _KERNEL).build()
            _state = {
                "ctx": ctx, "prog": prog, "queue": cl.CommandQueue(ctx),
                "bufs": {},   # npix -> 缓冲区组（保留少量，预览/全图来回切换不抖动）
                "desc": f"{dev.name.strip()} ({plat_name})",
            }
            print(f"[PhotoLab] GPU 加速已启用：{_state['desc']}", flush=True)
        except Exception as exc:
            print(f"[PhotoLab] GPU 加速不可用（{exc}），使用 CPU 多核", flush=True)
            _state = False
        return _state


def available() -> bool:
    return _init() is not False


def device_name() -> str:
    st = _init()
    return st["desc"] if st else ""


def process_simple(params, lin: np.ndarray) -> np.ndarray | None:
    """处理纯逐像素参数集（白平衡/基础/黑白/曲线/颜色分级），返回 uint8 RGB。

    不可用或失败时返回 None，由 core 回退 CPU 路径。
    """
    st = _init()
    if st is False:
        return None
    try:
        with _lock:
            return _run(st, params, lin)
    except Exception:
        global _fail_logged
        if not _fail_logged:
            _fail_logged = True
            traceback.print_exc()
            print("[PhotoLab] GPU 处理失败，本次已回退 CPU", flush=True)
        return None


def _basic_scalars(params) -> list:
    """基础影调的标量参数（顺序与内核一致，公式与 core.apply_basic 一一对应）。"""
    import core

    f32 = np.float32
    if params.temperature or params.tint:
        gains = core._wb_gains(params.temperature, params.tint)
        gr, gg, gb = (float(gains[0]), float(gains[1]), float(gains[2]))
    else:
        gr = gg = gb = 1.0
    return [
        f32(gr), f32(gg), f32(gb),
        f32(2.0 ** float(params.exposure)),
        f32(np.clip(params.contrast / 100.0, -1.0, 1.0)),
        f32(params.highlights / 100.0 * 1.8),
        f32(params.shadows / 100.0 * 0.6), f32(params.shadows / 100.0 * 0.10),
        f32(params.whites / 100.0 * 0.9), f32(params.whites / 100.0 * 0.06),
        f32(params.blacks / 100.0 * 0.09),
        f32(1.0 + params.saturation / 100.0),
        f32(params.vibrance / 100.0),
    ]


def _run(st, params, lin: np.ndarray) -> np.ndarray:
    import core

    h, w = lin.shape[:2]
    npix = h * w
    src = np.ascontiguousarray(lin, dtype=np.float32)

    # ---- 主机侧预计算 ----
    luts = core.channel_luts(params.curves)
    lut_arr = np.concatenate([luts["red"], luts["green"], luts["blue"]]) \
        .astype(np.float32)
    use_curves = 0 if core.curves_are_default(params.curves) else 1
    srgb_lut = np.ascontiguousarray(
        core._srgb_encode_lut().reshape(-1), dtype=np.uint8)

    exp = core.blend_exponent(params.blend)
    offs, exps, kinds = [], [], []
    for idx, name in enumerate(core.GRADING_RANGES):
        wl = params.wheels.get(name)
        if wl is None or not (wl.saturation or wl.luminance):
            continue
        vec = core.wheel_direction(wl.hue, wl.saturation / 100.0) * core.SAT_STRENGTH
        vec = vec + core.LUMA * np.float32(wl.luminance / 100.0 * core.LUM_STRENGTH)
        offs.append(vec)
        exps.append(exp)
        kinds.append(idx)
    if offs:
        off_arr = np.ascontiguousarray(np.stack(offs), np.float32)
        exp_arr = np.asarray(exps, np.float32)
        kind_arr = np.asarray(kinds, np.int32)
    else:
        off_arr = np.zeros((1, 3), np.float32)
        exp_arr = np.zeros(1, np.float32)
        kind_arr = np.zeros(1, np.int32)

    f32 = np.float32
    args = [
        np.int32(len(offs)), np.int32(core.LUT_SIZE), np.int32(use_curves),
        np.int32(w), np.int32(h),
        *_basic_scalars(params),
        f32(np.clip(params.blackwhite / 100.0, 0.0, 1.0)),   # 与 CPU 一致饱和到 [0,1]
    ]

    # ---- 缓冲区（按像素数复用）----
    import pyopencl as cl
    mf = cl.mem_flags
    bufs = st["bufs"]
    ent = bufs.get(npix)
    if ent is None:
        if len(bufs) >= 3:
            bufs.clear()
        ent = (
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=src.nbytes),
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=lut_arr.nbytes),
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=4 * 3 * 4),
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=4 * 4),
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=4 * 4),
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=65536),
            cl.Buffer(st["ctx"], mf.WRITE_ONLY, size=npix * 3),
        )
        bufs[npix] = ent
    b_src, b_lut, b_off, b_exp, b_kind, b_srgb, b_dst = ent

    q = st["queue"]
    out = np.empty((h, w, 3), np.uint8)
    cl.enqueue_copy(q, b_src, src)
    cl.enqueue_copy(q, b_lut, lut_arr)
    cl.enqueue_copy(q, b_off, off_arr)
    cl.enqueue_copy(q, b_exp, exp_arr)
    cl.enqueue_copy(q, b_kind, kind_arr)
    cl.enqueue_copy(q, b_srgb, srgb_lut)
    st["prog"].process_frame(
        q, ((npix + 255) // 256 * 256,), (256,),
        b_src, b_lut, b_off, b_exp, b_kind, b_srgb, *args, b_dst)
    cl.enqueue_copy(q, out, b_dst, is_blocking=True)
    return out


def process_basic_linear(params, lin: np.ndarray) -> np.ndarray | None:
    """只跑基础影调，返回 float32 线性光（不裁切）。失败返回 None。

    给「带效果算子」的渲染路径用：GPU 顶掉 CPU 的 apply_basic，
    效果/曲线/分级/蒙版仍按原有顺序在 CPU 上执行，因此画面语义不变。
    """
    st = _init()
    if st is False:
        return None
    try:
        with _lock:
            return _run_basic(st, params, lin)
    except Exception:
        global _fail_logged
        if not _fail_logged:
            _fail_logged = True
            traceback.print_exc()
            print("[PhotoLab] GPU 基础影调失败，本次已回退 CPU", flush=True)
        return None


def _run_basic(st, params, lin: np.ndarray) -> np.ndarray:
    h, w = lin.shape[:2]
    npix = h * w
    src = np.ascontiguousarray(lin, dtype=np.float32)

    import pyopencl as cl
    mf = cl.mem_flags
    bufs = st.setdefault("bufs_f32", {})
    ent = bufs.get(npix)
    if ent is None:
        if len(bufs) >= 3:
            bufs.clear()
        ent = (
            cl.Buffer(st["ctx"], mf.READ_ONLY, size=src.nbytes),
            cl.Buffer(st["ctx"], mf.WRITE_ONLY, size=src.nbytes),
        )
        bufs[npix] = ent
    b_src, b_dst = ent

    q = st["queue"]
    out = np.empty((h, w, 3), np.float32)
    cl.enqueue_copy(q, b_src, src)
    st["prog"].basic_frame(
        q, ((npix + 255) // 256 * 256,), (256,),
        b_src, b_dst, np.int32(w), np.int32(h), *_basic_scalars(params))
    cl.enqueue_copy(q, out, b_dst, is_blocking=True)
    return out
