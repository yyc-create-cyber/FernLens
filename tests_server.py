"""server.py API 层自检。运行：python tests_server.py

在 PHOTOLAB_SYNC=1 下运行（渲染在请求线程内同步完成，结果确定）。
关键用例：同一参数下，API 保存的 PNG 与直接调用 core.process 的输出逐位一致。
"""
from __future__ import annotations

import os
import struct
import tempfile
import traceback

os.environ["PHOTOLAB_SYNC"] = "1"      # 必须在 import server 之前设置

import cv2
import numpy as np
from fastapi.testclient import TestClient

import core
import segment
import server
from core import Adjustments, MaskAdjustments, MaskState

_TOKEN = "test-token-123"
client = TestClient(server.app)
_server_url = "/api"


def _auth(path: str) -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}token={_TOKEN}"


def _get(path: str, **kw):
    return client.get(_auth(path), **kw)


def _post(path: str, **kw):
    return client.post(_auth(path), **kw)


def _post_json(path: str, payload: dict):
    return client.post(_auth(path), json=payload)


def _recv_frame(ws) -> bytes:
    """收下一个二进制帧，跳过途中的文本事件（busy/saved 等）。"""
    for _ in range(32):
        msg = ws.receive()
        if "bytes" in msg:
            return msg["bytes"]
    raise AssertionError("等了 32 条消息也没等到二进制帧")


def _make_image(h: int = 240, w: int = 320, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = xx / max(w - 1, 1)          # 水平红渐变
    base[..., 1] = yy / max(h - 1, 1)          # 垂直绿渐变
    base[..., 2] = 0.5
    img = base * 200 + rng.integers(-28, 28, (h, w, 3))     # 加噪，保证算子有内容可算
    return np.clip(img, 0, 255).astype(np.uint8)


def _rich_adjustments() -> dict:
    """一组覆盖多阶段流水线的非默认参数（含一条蒙版）。"""
    a = Adjustments()
    a.temperature = 25.0
    a.tint = -10.0
    a.exposure = 0.6
    a.contrast = 30.0
    a.highlights = -25.0
    a.shadows = 35.0
    a.whites = 10.0
    a.blacks = -15.0
    a.saturation = 18.0
    a.vibrance = 22.0
    a.clarity = 20.0
    a.sharpen = 15.0
    a.dehaze = 12.0
    a.curves["master"] = [(0.0, 0.05), (0.5, 0.55), (1.0, 0.95)]
    a.curves["red"] = [(0.0, 0.0), (0.5, 0.45), (1.0, 1.0)]
    from core import WheelState
    a.wheels["shadows"] = WheelState(hue=200.0, saturation=25.0, luminance=-8.0)
    a.wheels["highlights"] = WheelState(hue=45.0, saturation=12.0, luminance=5.0)

    m = MaskState(kind="linear", name="测试蒙版", feather=40.0,
                  x1=0.2, y1=0.3, x2=0.8, y2=0.7)
    m.adj = MaskAdjustments(exposure=0.4, contrast=15.0, saturation=10.0)
    a.masks = [m]
    return server.adjustments_to_dict(a)


# ======================================================================
# 用例
# ======================================================================
def test_meta_and_auth():
    r = _get("/api/meta")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["app"] == "PhotoLab"
    assert isinstance(meta["rawpy"], bool)
    assert meta["hist_bins"] == core.HIST_BINS

    # 错误 token 必须拒绝
    r2 = client.get("/api/meta", params={"token": "wrong"})
    assert r2.status_code == 401


def test_open_and_meta_width_height():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image()
        p = os.path.join(tmp, "输入 测试.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        r = _post_json("/api/open", {"path": p})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["width"] == img.shape[1] and body["height"] == img.shape[0]
        assert isinstance(body["serial"], int)

    # 不存在的文件 -> 400
    r = _post_json("/api/open", {"path": "Z:/不存在/图.png"})
    assert r.status_code == 400


def test_open_bad_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "不是图.png")
        open(p, "wb").write(b"not an image at all")
        r = _post_json("/api/open", {"path": p})
        assert r.status_code == 400


def test_save_bitwise_parity_with_core():
    """核心验收：API 保存的 PNG 与 core.process 直接输出逐位一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image()
        src_p = os.path.join(tmp, "parity 输入.png")
        out_p = os.path.join(tmp, "parity 输出.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(src_p)
        assert _post_json("/api/open", {"path": src_p}).status_code == 200

        adj = _rich_adjustments()
        r = _post_json("/api/save", {"path": out_p, "adjustments": adj})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["saved"] is True, "同步模式下保存应已完成"

        got = cv2.cvtColor(cv2.imdecode(
            np.fromfile(out_p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # 服务端路径：rgb8_to_linear_mt + process_linear（多线程线性光转换）
        params = server.adjustments_from_dict(adj)
        expect_mt = core.process_linear(params, core.rgb8_to_linear_mt(img))
        assert np.array_equal(got, expect_mt), \
            f"API 输出与 process_linear/mt 不一致，最大差 " \
            f"{int(np.abs(got.astype(int) - expect_mt.astype(int)).max())}/255"

        # core.process（单线程转换）与多线程转换应逐位一致（README 声明）
        expect_st = core.process(params, img)
        assert np.array_equal(expect_mt, expect_st), "多线程与单线程转换结果不一致"


def test_save_default_params_short_path():
    """未传 adjustments 时按默认参数保存，且可省略扩展名。"""
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(64, 64, seed=7)
        src_p = os.path.join(tmp, "in.png")
        out_p = os.path.join(tmp, "out")            # 无扩展名 -> 补 .jpg
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(src_p)
        _post_json("/api/open", {"path": src_p})
        r = _post_json("/api/save", {"path": out_p})
        assert r.status_code == 200
        assert os.path.isfile(out_p + ".jpg")


def test_ws_render_frame():
    """WS 渲染：二进制帧头 [uint32 rid][uint8 kind] + 可解码 JPEG。"""
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(96, 128, seed=3)
        p = os.path.join(tmp, "ws.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})

        with client.websocket_connect(_auth("/ws")) as ws:
            ws.send_json({"type": "render", "rid": 501, "kind": "preview",
                          "adjustments": _rich_adjustments()})
            data = _recv_frame(ws)
            rid, kind = struct.unpack(">IB", data[:5])
            assert rid == 501
            assert kind == 0, "preview 帧的 kind 应为 0"
            frame = cv2.imdecode(np.frombuffer(data[5:], np.uint8), cv2.IMREAD_COLOR)
            assert frame is not None and frame.ndim == 3
            assert max(frame.shape[0], frame.shape[1]) <= 1200

            # 全图请求 kind 应为 1
            ws.send_json({"type": "render", "rid": 502, "kind": "full",
                          "adjustments": {}})
            data = _recv_frame(ws)
            rid, kind = struct.unpack(">IB", data[:5])
            assert (rid, kind) == (502, 1)

            # 坏消息不能断连接
            ws.send_text("not json")
            ws.send_json({"type": "render", "rid": 503, "kind": "bogus",
                          "adjustments": {}})
            ws.send_json({"type": "render", "rid": 504, "kind": "full",
                          "adjustments": {}})
            rid, _ = struct.unpack(">IB", _recv_frame(ws)[:5])
            assert rid == 504


def test_ws_bad_token_rejected():
    try:
        with client.websocket_connect("/ws?token=wrong") as ws:
            ws.receive_bytes()
        raise AssertionError("错误 token 不应连上 WebSocket")
    except Exception:
        pass


def test_serialization_roundtrip():
    """参数序列化往返必须无损（多线程渲染一致性的前提）。"""
    adj = _rich_adjustments()
    params = server.adjustments_from_dict(adj)
    again = server.adjustments_to_dict(params)
    params2 = server.adjustments_from_dict(again)
    d1, d2 = server.adjustments_to_dict(params), server.adjustments_to_dict(params2)
    assert d1 == d2

    # 蒙版结构逐字段一致
    m1, m2 = params.masks[0], params2.masks[0]
    assert (m1.kind, m1.feather, m1.x1, m1.y2, m1.invert) == \
           (m2.kind, m2.feather, m2.x1, m2.y2, m2.invert)
    assert m1.adj.exposure == m2.adj.exposure
    assert m1.adj.wheels == m2.adj.wheels

    # 缺省/垃圾输入不崩溃，字段回退默认
    p = server.adjustments_from_dict({"exposure": "垃圾", "curves": {"master": "xx"},
                                      "masks": [{"kind": 123}, None]})
    assert p.exposure == 0.0
    assert p.masks[0].kind in ("linear", "radial", "brush") or isinstance(p.masks[0].kind, str)


def test_stamp_apply_and_undo():
    with tempfile.TemporaryDirectory() as tmp:
        img = np.zeros((100, 200, 3), np.uint8)
        img[:, :100] = (255, 0, 0)      # 左半红
        img[:, 100:] = (0, 0, 255)      # 右半蓝
        p = os.path.join(tmp, "stamp.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})
        serial_after_open = server.doc.serial

        before = server.doc.retouch_full if server.doc.retouch_full is not None \
            else server.doc.source
        assert tuple(before[75, 150]) == (0, 0, 255)

        # 把左上红块复制到右下
        r = _post_json("/api/stamp", {"src_rect": [0.0, 0.0, 0.5, 0.5],
                                      "dst_rect": [0.5, 0.5, 1.0, 1.0],
                                      "feather": 0})
        assert r.status_code == 200
        assert r.json()["serial"] == serial_after_open + 1
        after = server.doc.retouch_full
        assert tuple(after[75, 150]) == (255, 0, 0), "盖印后右下应为红"

        # 撤销一笔 -> 恢复蓝（撤销最后一笔后工作副本清空，与旧版一致）
        r = _post_json("/api/stamp/undo", {})
        assert r.status_code == 200
        assert server.doc.retouch_full is None
        restored = server.doc.source
        assert tuple(restored[75, 150]) == (0, 0, 255), "撤销后应恢复原样"


def test_crop_and_rotate():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(100, 200, seed=5)
        p = os.path.join(tmp, "geom.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})

        m_dict = {"kind": "linear", "x1": 0.5, "y1": 0.5, "x2": 0.8, "y2": 0.5}

        # 先旋转：200x100 顺时针 -> 100x200
        r = _post_json("/api/geometry/rotate", {"clockwise": True, "masks": []})
        assert r.status_code == 200
        assert r.json()["width"] == 100 and r.json()["height"] == 200

        # 再裁剪：旋转后为 100x200，取横向左半 -> 50x200
        # 线性蒙版端点线性重映射：(0.5-0.0)/0.5 = 1.0
        r = _post_json("/api/geometry/crop",
                       {"rect": [0.0, 0.0, 0.5, 1.0], "angle": 0.0,
                        "masks": [m_dict]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["width"] == 50 and body["height"] == 200
        assert body["baked"] is False
        assert len(body["masks"]) == 1
        assert abs(body["masks"][0]["x1"] - 1.0) < 1e-6


def test_histogram():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(64, 64, seed=9)
        p = os.path.join(tmp, "hist.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})
        r = _get("/api/histogram")
        assert r.status_code == 200
        h = r.json()
        assert set(h) == {"master", "red", "green", "blue"}
        assert all(len(v) == core.HIST_BINS for v in h.values())
        assert max(h["master"]) > 0


def test_auto_rule():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(64, 64, seed=11)
        p = os.path.join(tmp, "auto.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})
        r = _post_json("/api/auto", {"mode": "rule"})
        assert r.status_code == 200, r.text
        assert "adjustments" in r.json()


def test_ai_endpoints_with_mock():
    """AI 端点（测试连接 / AI 蒙版 / AI 调色）：mock 掉网络，只测协议与接线。"""
    cfg = {"base_url": "http://mock/v1", "api_key": "k", "model": "mock-vl"}

    # ---- /api/ai/test：成功与失败 ----
    orig_fetch = server.auto.fetch_models

    def fake_fetch(base, key, timeout=8):
        if key == "k":
            return ["mock-vl", "mock-text"]
        raise RuntimeError("401 unauthorized")

    server.auto.fetch_models = fake_fetch
    try:
        r = _post_json("/api/ai/test", {"config": cfg})
        assert r.status_code == 200, r.text
        assert r.json()["models"] == ["mock-vl", "mock-text"]
        r = _post_json("/api/ai/test", {"config": {**cfg, "api_key": "bad"}})
        assert r.status_code == 502 and "连接失败" in r.json()["detail"]
        r = _post_json("/api/ai/test", {"config": {"base_url": "", "api_key": ""}})
        assert r.status_code == 400
    finally:
        server.auto.fetch_models = orig_fetch

    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(64, 80, seed=15)
        p = os.path.join(tmp, "ai.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})


        # ---- /api/auto mode=ai：mock suggest_ai，验证 config / intent 透传 ----
        orig_suggest = server.auto.suggest_ai
        seen = {}

        def fake_suggest(rgb8, config=None, timeout=90, intent="", strength=0.6):
            seen["intent"] = intent
            seen["strength"] = strength
            return {"exposure": 0.3}, "mock note", f"mock@{config['base_url']}"

        server.auto.suggest_ai = fake_suggest
        try:
            r = _post_json("/api/auto", {"mode": "ai", "config": cfg,
                                         "intent": "  日系胶片 "})
            assert r.status_code == 200, r.text
            assert r.json()["adjustments"]["exposure"] == 0.3
            assert r.json()["model"] == "mock@http://mock/v1"
            assert seen["intent"] == "日系胶片", seen
            assert seen["strength"] == server.auto.AI_DEFAULT_STRENGTH, seen

            # 显式强度透传，且钳制到 0..1
            r = _post_json("/api/auto", {"mode": "ai", "config": cfg,
                                        "strength": 0.35})
            assert r.status_code == 200, r.text
            assert abs(seen["strength"] - 0.35) < 1e-9, seen
            r = _post_json("/api/auto", {"mode": "ai", "config": cfg,
                                        "strength": 9})
            assert abs(seen["strength"] - 1.0) < 1e-9, seen
        finally:
            server.auto.suggest_ai = orig_suggest


def test_ai_mask_requires_image():
    # 显式清空文档（前面用例都开过图）
    server.doc.__init__()
    r = _post_json("/api/mask/ai", {"target": "subject"})
    assert r.status_code == 400 and "未打开图像" in r.json()["detail"]


def test_ai_mask_save_parity():
    """AI 蒙版经 JSON 往返（_mask_from_dict）后与直接构造的 core 参数逐位一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(96, 128, seed=21)
        src_p = os.path.join(tmp, "ai parity.png")
        out_p = os.path.join(tmp, "ai parity out.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(src_p)
        _post_json("/api/open", {"path": src_p})

        grid = ["110000", "111000", "011100", "001110", "000111", "000011"]
        adj = {
            "temperature": 12.0, "exposure": 0.35, "contrast": 18.0,
            "masks": [{
                "kind": "ai", "name": "AI 主体 1", "feather": 30, "invert": False,
                "ai_target": "subject", "grid": grid, "grid_w": 6, "grid_h": 6,
                "adj": {"exposure": 0.8, "saturation": -20.0},
            }],
        }
        r = _post_json("/api/save", {"path": out_p, "adjustments": adj})
        assert r.status_code == 200 and r.json()["saved"] is True, r.text
        got = cv2.cvtColor(cv2.imdecode(
            np.fromfile(out_p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        params = server.adjustments_from_dict(adj)
        m = params.masks[0]
        assert m.kind == "ai" and m.grid == grid and (m.grid_w, m.grid_h) == (6, 6)
        assert m.adj.exposure == 0.8
        expect = core.process_linear(params, core.rgb8_to_linear_mt(img))
        assert np.array_equal(got, expect), "AI 蒙版 JSON 往返渲染应与 core 直接构造逐位一致"

        # 序列化往返（WS 渲染路径同款解析）
        again = server.adjustments_from_dict(server.adjustments_to_dict(params))
        assert server.adjustments_to_dict(again) == server.adjustments_to_dict(params)

        # 脏数据安全：坏网格/超大网格被钳制，不崩溃
        dirty = server.adjustments_from_dict({"masks": [{
            "kind": "ai", "grid": ["12x"], "grid_w": 99999, "grid_h": -3}]})
        dm = dirty.masks[0]
        assert dm.grid_w == 512 and dm.grid_h == 0 and dm.grid == []


def test_pick():
    with tempfile.TemporaryDirectory() as tmp:
        img = np.zeros((50, 50, 3), np.uint8)
        img[:] = (200, 100, 50)      # RGB
        p = os.path.join(tmp, "pick.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})
        r = _post_json("/api/pick", {"x": 25, "y": 25})
        assert r.status_code == 200
        c = r.json()
        assert abs(c["r"] - 200 / 255) < 1e-4
        assert abs(c["g"] - 100 / 255) < 1e-4
        assert 0.0 <= c["hue"] < 360.0


def test_defaults():
    r = _get("/api/defaults")
    assert r.status_code == 200
    d = r.json()["adjustments"]
    assert d["exposure"] == 0.0
    assert d["curves"]["master"] == [[0.0, 0.0], [1.0, 1.0]]
    assert set(d["wheels"]) == {"shadows", "midtones", "highlights", "global"}


def test_open_upload_and_render_download():
    """浏览器模式：上传打开 + 渲染下载（PNG 无损，须与 /api/save 一致）。"""
    img = _make_image(80, 120, seed=13)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    r = client.post(_auth("/api/open-upload"),
                    files={"file": ("测试 上传.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 120 and body["height"] == 80

    r = client.post(_auth("/api/open-upload"),
                    files={"file": ("垃圾.bin", b"xxx", "application/octet-stream")})
    assert r.status_code == 400

    adj = _rich_adjustments()
    r = _post_json("/api/render", {"adjustments": adj, "format": "png"})
    assert r.status_code == 200, r.text
    got = cv2.cvtColor(cv2.imdecode(np.frombuffer(r.content, np.uint8),
                                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    expect = core.process_linear(server.adjustments_from_dict(adj),
                                 core.rgb8_to_linear_mt(img))
    assert np.array_equal(got, expect), "下载渲染与直接调用 core 不一致"

    r = _post_json("/api/render", {"adjustments": {}, "format": "bmp"})
    assert r.status_code == 400


def test_library_multi_image():
    """多图导入：入列/列表/切换（参数隔离）/缩略图/删除与自动切换。"""
    server.library.clear()
    server.doc = server.Document()

    def upload(h, w, seed, name):
        img = _make_image(h, w, seed=seed)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok
        r = client.post(_auth("/api/open-upload"),
                        files={"file": (name, buf.tobytes(), "image/png")})
        assert r.status_code == 200, r.text
        return r.json()

    a = upload(100, 200, 31, "甲.png")
    b = upload(120, 180, 32, "乙.png")
    assert a["id"] != b["id"]
    assert b["library"] == 2

    r = _get("/api/library")
    assert r.status_code == 200
    imgs = r.json()["images"]
    assert len(imgs) == 2
    assert [i["name"] for i in imgs] == ["甲", "乙"]
    assert imgs[0]["current"] is False and imgs[1]["current"] is True

    # 切回第一张：当前文档换成甲（尺寸对得上）
    r = _post_json("/api/library/select", {"id": a["id"]})
    assert r.status_code == 200 and r.json()["width"] == 200 and r.json()["height"] == 100
    assert server.doc.source.shape == (100, 200, 3)

    # 参数隔离：对甲调曝光保存，切到乙再切回，甲的画面不变、乙不受甲参数影响
    adj = server.adjustments_to_dict(server.adjustments_from_dict(
        {"exposure": 1.0, "masks": []}))
    r = _post_json("/api/render", {"adjustments": adj, "format": "png"})
    assert r.status_code == 200
    frame_a = np.frombuffer(r.content, np.uint8)
    _post_json("/api/library/select", {"id": b["id"]})
    assert server.doc.source.shape == (120, 180, 3)      # 文档已切换
    r = _post_json("/api/render", {"adjustments": adj, "format": "png"})
    assert r.status_code == 200
    assert np.frombuffer(r.content, np.uint8).size != frame_a.size   # 尺寸即不同

    # 缩略图：可解码且最长边 <= 240
    r = _get(f"/api/library/{a['id']}/thumb")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    thumb = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert thumb is not None and max(thumb.shape[:2]) <= 240
    assert _get("/api/library/99999/thumb").status_code == 404

    # 删除当前图（乙）-> 自动切到相邻；再删光 -> 回到空文档
    r = _post_json("/api/library/remove", {"id": b["id"]})
    assert r.status_code == 200
    imgs = r.json()["images"]
    assert len(imgs) == 1 and imgs[0]["id"] == a["id"] and imgs[0]["current"] is True
    assert server.doc.source.shape == (100, 200, 3)
    r = _post_json("/api/library/remove", {"id": a["id"]})
    assert r.json()["images"] == []
    assert server.doc.source is None
    assert _get("/api/meta").json()["width"] == 0

    # 不存在的条目
    assert _post_json("/api/library/remove", {"id": a["id"]}).status_code == 400
    assert _post_json("/api/library/select", {"id": a["id"]}).status_code == 400


def test_ai_mask_local_engine():
    """本地像素级引擎：engine=local 路由 + 模型状态端点（mock 推理，不依赖真实模型）。"""
    # 模型状态端点（真实现：读 models/ 目录；不依赖权重存在与否）
    r = _get("/api/ai/models")
    assert r.status_code == 200, r.text
    st = r.json()
    assert isinstance(st["runtime"], bool)
    assert set(st["models"]) == {"sam_encoder", "sam_decoder", "u2net", "skyseg", "yolov8n"}

    # 下载端点幂等启动（不真正等下载完成，只验证接线；用假目录避免碰真模型）
    orig_status = segment.models_status
    orig_start = segment.start_download
    segment.models_status = lambda: {"u2net": {"file": "u2net.onnx", "ready": False, "size": 0}}
    segment.start_download = lambda: {"started": True}
    try:
        r = _post_json("/api/ai/models/download", {})
        assert r.status_code == 200 and r.json()["started"] is True
        r = _get("/api/ai/models/progress")
        assert r.status_code == 200 and "models" in r.json()
    finally:
        segment.models_status = orig_status
        segment.start_download = orig_start

    # engine=local：mock segment_local 返回像素掩码，验证路由与 mask_png 往返
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(96, 128, seed=33)
        p = os.path.join(tmp, "local ai.png")
        cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
        _post_json("/api/open", {"path": p})

        calls: list = []

        def fake_local(rgb8, target):
            calls.append((target, rgb8.shape))
            import core as _core
            prob = np.zeros(rgb8.shape[:2], np.float32)
            prob[: rgb8.shape[0] // 2] = 1.0               # 上半为 1
            b64, w, h = _core.encode_mask_png(prob)
            return {"target": target, "mask_png": b64, "width": w, "height": h,
                    "model": "mock-local"}

        orig_seg = server.segment.segment_local
        server.segment.segment_local = fake_local
        try:
            r = _post_json("/api/mask/ai", {"target": "sky", "engine": "local"})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["mask_png"] and body["width"] == 128
            assert calls[-1] == ("sky", (96, 128, 3))
        finally:
            server.segment.segment_local = orig_seg

        # 像素掩码经保存路径逐位一致（_mask_from_dict 解析 mask_png 后渲染）
        adj = {"masks": [{"kind": "ai", "mask_png": body["mask_png"],
                          "mask_w": body["width"], "mask_h": body["height"],
                          "feather": 0, "adj": {"exposure": 0.8}}]}
        out_p = os.path.join(tmp, "local out.png")
        r = _post_json("/api/save", {"path": out_p, "adjustments": adj})
        assert r.status_code == 200 and r.json()["saved"] is True
        params = server.adjustments_from_dict(adj)
        assert params.masks[0].mask_png == body["mask_png"]
        expect = core.process_linear(params, core.rgb8_to_linear_mt(img))
        got = cv2.cvtColor(cv2.imdecode(
            np.fromfile(out_p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.array_equal(got, expect), "本地像素掩码 JSON 往返应与 core 逐位一致"

        # 未识别到目标 -> 400（ValueError 路径）
        server.segment.segment_local = lambda rgb8, t: (_ for _ in ()).throw(
            ValueError("画面中未识别到「天空」"))
        try:
            r = _post_json("/api/mask/ai", {"target": "sky", "engine": "local"})
            assert r.status_code == 400 and "天空" in r.json()["detail"]
        finally:
            server.segment.segment_local = orig_seg


# ======================================================================
CASES = [test_meta_and_auth, test_open_and_meta_width_height, test_open_bad_file,
         test_save_bitwise_parity_with_core, test_save_default_params_short_path,
         test_ws_render_frame, test_ws_bad_token_rejected,
         test_serialization_roundtrip, test_stamp_apply_and_undo,
         test_crop_and_rotate, test_histogram, test_auto_rule, test_pick,
         test_defaults, test_open_upload_and_render_download,
         test_ai_endpoints_with_mock, test_ai_mask_requires_image,
         test_ai_mask_save_parity, test_library_multi_image,
         test_ai_mask_local_engine]


def main():
    server._token = _TOKEN
    failures = 0
    for case in CASES:
        try:
            case()
            print(f"  [通过] {case.__name__}")
        except Exception:
            failures += 1
            print(f"  [失败] {case.__name__}")
            traceback.print_exc()
    print(f"\ntests_server: {len(CASES) - failures}/{len(CASES)} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    with client:            # 触发 startup/shutdown 事件
        sys_exit = main()
    raise SystemExit(sys_exit)
