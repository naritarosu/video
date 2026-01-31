# generate.py
import base64
import io
import os
import re
import time
import subprocess
import signal
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests
from PIL import Image


# =========================
# reForge (A1111 API) helpers
# =========================
def api_get(base_url: str, path: str, timeout: int = 60) -> Any:
    url = base_url.rstrip("/") + path
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise gr.Error("reForge (127.0.0.1:7860) に接続できません。Start reForge を押して起動してください。")


def api_post(base_url: str, path: str, payload: Dict[str, Any], timeout: int = 600) -> Any:
    url = base_url.rstrip("/") + path
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise gr.Error("reForge (127.0.0.1:7860) に接続できません。Start reForge を押して起動してください。")


def list_sd_models(base_url: str) -> List[str]:
    models = api_get(base_url, "/sdapi/v1/sd-models", timeout=60)
    return [m["title"] for m in models if "title" in m]


def set_sd_model(base_url: str, model_title: str) -> None:
    api_post(base_url, "/sdapi/v1/options", {"sd_model_checkpoint": model_title}, timeout=120)


def b64_to_pil(b64png: str) -> Image.Image:
    raw = base64.b64decode(b64png)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# =========================
# Gradio actions
# =========================
@dataclass
class ImgGenOut:
    image: Image.Image
    used_prompt: str
    info: str


def build_full_prompt(prefix: str, character: str, scene_prompt: str) -> str:
    prefix = (prefix or "").strip().strip(",")
    character = (character or "").strip().strip(",")
    scene_prompt = (scene_prompt or "").strip()
    parts = [p for p in [prefix, character, scene_prompt] if p]
    return ", ".join(parts)


def generate_image(
    base_url: str,
    model_title: str,
    prefix: str,
    character: str,
    scene_prompt: str,
    negative: str,
    steps: int,
    width: int,
    height: int,
    cfg_scale: float,
    seed: int,
) -> Tuple[Image.Image, str, str]:
    if model_title:
        set_sd_model(base_url, model_title)

    prompt = build_full_prompt(prefix, character, scene_prompt)

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative or "",
        "steps": int(steps),
        "width": int(width),
        "height": int(height),
        "cfg_scale": float(cfg_scale),
        "seed": int(seed),
    }

    res = api_post(base_url, "/sdapi/v1/txt2img", payload, timeout=1200)
    img = b64_to_pil(res["images"][0])
    info = res.get("info", "")
    return img, prompt, info


# =========================
# reForge process control (optional)
# =========================
DEFAULT_PREFIX = "photo anime, masterpiece, high quality, absurdres"
DEFAULT_NEG = (
    "watermark, signature, text, logo, bad anatomy, extra digit, fewer digits, worst quality, "
    "crowd, group, multiple people, extra people, background people, background characters, bystanders, "
    "pedestrians, silhouettes, faces, two people"
)

REFORGE_DIR = "/home/akane/Documents/crawler/story/stable-diffusion-webui-reForge"
REFORGE_PORT = 7860

STATE_DIR = Path("outputs")
STATE_DIR.mkdir(exist_ok=True)
REFORGE_PIDFILE = STATE_DIR / "reforge.pid"
REFORGE_LOG = STATE_DIR / "reforge.log"

REFORGE_START_CMD = (
    f'cd "{REFORGE_DIR}" && '
    'export COMMANDLINE_ARGS="--api --listen 127.0.0.1 --port 7860" && '
    "bash webui.sh"
)


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_reforge_running(timeout_sec: float = 120.0) -> str:
    """
    app.py 起動時に呼ぶ。
    - すでに port が開いていれば何もしない
    - 開いていなければ start_reforge() して、port open まで待つ
    """
    if _port_open("127.0.0.1", REFORGE_PORT):
        return "reForge already running."

    msg = start_reforge()
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if _port_open("127.0.0.1", REFORGE_PORT):
            return f"{msg} | READY"
        time.sleep(0.5)

    return f"{msg} | TIMEOUT (port {REFORGE_PORT} not open yet). Check outputs/reforge.log"


def reforge_status() -> str:
    return "RUNNING (port 7860 open)" if _port_open("127.0.0.1", REFORGE_PORT) else "STOPPED"


def start_reforge() -> str:
    if _port_open("127.0.0.1", REFORGE_PORT):
        return "reForge is already running."

    with open(REFORGE_LOG, "ab") as lf:
        p = subprocess.Popen(
            ["bash", "-lc", REFORGE_START_CMD],
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    REFORGE_PIDFILE.write_text(str(p.pid))
    return f"Started reForge (pid={p.pid})."


def _find_pids_by_port(port: int) -> List[int]:
    pids: List[int] = []
    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"lsof -t -iTCP:{port} -sTCP:LISTEN"], text=True
        ).strip()
        if out:
            pids = [int(x) for x in out.splitlines() if x.strip().isdigit()]
            if pids:
                return sorted(set(pids))
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"fuser -n tcp {port} 2>/dev/null"], text=True
        ).strip()
        if out:
            nums = re.findall(r"\b\d+\b", out)
            pids = [int(x) for x in nums]
            if pids:
                return sorted(set(pids))
    except Exception:
        pass

    try:
        out = subprocess.check_output(["bash", "-lc", f"ss -lptn 'sport = :{port}'"], text=True)
        nums = re.findall(r"pid=(\d+)", out)
        pids = [int(x) for x in nums]
        if pids:
            return sorted(set(pids))
    except Exception:
        pass

    return []


def _send_signal_to_pid(pid: int, sig: signal.Signals) -> None:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    except Exception:
        pass

    try:
        os.kill(pid, sig)
    except Exception:
        pass


def _stop_by_port(timeout_sec: float) -> str:
    if not _port_open("127.0.0.1", REFORGE_PORT):
        return "reForge is not running."

    def _current_pids():
        return _find_pids_by_port(REFORGE_PORT)

    pids = _current_pids()
    if not pids:
        return "PID not managed and could not find PID by port."

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        pids = _current_pids()
        if not pids:
            return "Stopped reForge (by port)."

        for p in pids:
            _send_signal_to_pid(p, sig)

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if not _port_open("127.0.0.1", REFORGE_PORT):
                return f"Stopped reForge (by port, pids={pids}, sig={sig.name})."
            time.sleep(0.2)

    pids = _current_pids()
    return f"Tried to stop unmanaged reForge (pids={pids}) but port is still open. Check process/log."


def stop_reforge(timeout_sec: float = 30.0) -> str:
    pid = None
    if REFORGE_PIDFILE.exists():
        try:
            pid = int(REFORGE_PIDFILE.read_text().strip())
        except Exception:
            pid = None

    if pid is not None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            _send_signal_to_pid(pid, sig)
            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                if not _port_open("127.0.0.1", REFORGE_PORT):
                    try:
                        REFORGE_PIDFILE.unlink()
                    except Exception:
                        pass
                    return f"Stopped reForge ({sig.name}, managed pid)."
                time.sleep(0.2)

        try:
            REFORGE_PIDFILE.unlink()
        except Exception:
            pass
        return _stop_by_port(timeout_sec)

    return _stop_by_port(timeout_sec)


# =========================
# UI builder: returns handles for integration
# =========================
def build_generate_tab() -> Dict[str, Any]:
    """
    Creates the "Generate" UI inside current Blocks/Tab context.
    Returns component handles needed by app.py (e.g., scene_prompt textbox).
    """
    gr.Markdown("## 画像生成（reForge）")

    with gr.Row():
        base_url = gr.Textbox(label="reForge API Base URL", value="http://127.0.0.1:7860")

    gr.Markdown("### reForge コントロール（任意）")
    with gr.Row():
        btn_status = gr.Button("Status")
        btn_start = gr.Button("Start reForge")
        btn_stop = gr.Button("Stop reForge")

    reforge_state = gr.Textbox(label="reForge state", value=reforge_status(), lines=1)

    def _ui_status():
        return reforge_status()

    def _ui_start():
        msg = start_reforge()
        return f"{msg} | {reforge_status()}"

    def _ui_stop():
        msg = stop_reforge()
        return f"{msg} | {reforge_status()}"

    btn_status.click(_ui_status, inputs=[], outputs=[reforge_state])
    btn_start.click(_ui_start, inputs=[], outputs=[reforge_state])
    btn_stop.click(_ui_stop, inputs=[], outputs=[reforge_state])

    refresh_btn = gr.Button("Refresh models")
    sd_model = gr.Dropdown(label="SD model (reForge)", choices=[], value=None, interactive=True)

    def _refresh_models(url):
        if not _port_open("127.0.0.1", REFORGE_PORT):
            raise gr.Error("reForge が停止中です。Start reForge を押してから Refresh models を押してください。")
        models = list_sd_models(url)
        return gr.Dropdown(choices=models, value=models[0] if models else None)

    refresh_btn.click(_refresh_models, inputs=[base_url], outputs=[sd_model])

    gr.Markdown("---")

    with gr.Row():
        prefix = gr.Textbox(label="Prompt Prefix（デフォルト）", value=DEFAULT_PREFIX)
        negative = gr.Textbox(label="Negative Prompt（デフォルト）", value=DEFAULT_NEG)

    character = gr.Textbox(
        label="主人公の見た目（手動入力・別枠）",
        value="(character description here)",
    )

    scene_prompt = gr.Textbox(label="画像生成プロンプト（Story Plannerから送られる/コピペ）", lines=5)

    with gr.Row():
        steps = gr.Slider(label="Steps", minimum=1, maximum=60, value=20, step=1)
        cfg = gr.Slider(label="CFG scale", minimum=1, maximum=12, value=7, step=0.5)
        seed = gr.Number(label="Seed (-1 random)", value=-1, precision=0)

    with gr.Row():
        width = gr.Number(label="Width", value=1280, precision=0)
        height = gr.Number(label="Height", value=720, precision=0)

    gen_btn = gr.Button("Generate Image", variant="primary")

    with gr.Row():
        out_img = gr.Image(label="Generated image", type="pil")
        out_info = gr.Textbox(label="reForge info", lines=8)

    used_prompt = gr.Textbox(label="実際に投げた最終プロンプト（prefix + 主人公 + scene）", lines=2)

    def _on_generate_image(url, model, pfx, ch, sp, neg, st, w, h, cfg_scale, sd):
        img, final_prompt, info = generate_image(
            url, model, pfx, ch, sp, neg, st, int(w), int(h), float(cfg_scale), int(sd)
        )
        return img, info, final_prompt

    gen_btn.click(
        _on_generate_image,
        inputs=[base_url, sd_model, prefix, character, scene_prompt, negative, steps, width, height, cfg, seed],
        outputs=[out_img, out_info, used_prompt],
    )

    return {
        "scene_prompt": scene_prompt,  # app.py がここに流し込む
        "character": character,
        "prefix": prefix,
        "negative": negative,
    }


# generate.py 単体起動も可能にしておく（デバッグ用）
def main():
    with gr.Blocks(title="Generate UI (Image only)") as app:
        build_generate_tab()
    app.queue().launch(server_name="127.0.0.1", server_port=7861, show_error=True)


if __name__ == "__main__":
    main()
