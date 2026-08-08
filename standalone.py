#!/usr/bin/env python3
"""GapFiller standalone - use it without ComfyUI.

Command line:
    gapfiller in.mp4 out.mp4 --fps 60
    gapfiller in.mp4 out.mp4 --multiplier 4 --sharpness 1.1

Drag-and-drop web UI (needs `pip install gradio`):
    gapfiller-ui

Only ffmpeg and PyTorch are required for the CLI.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gapfiller_model import GapFillerNet

CFG = {
    "current": ((128, 96, 72), (8, 6, 4)),
    "bigger": ((192, 144, 96), (10, 8, 6)),
    "large": ((256, 192, 128), (12, 10, 8)),
}
WEIGHT_URLS = ["https://github.com/KernelPanicKitten/ComfyUI-GapFiller/releases/download/v0.1.0/gapfiller_v1.pt"]


def default_ckpt_dir():
    return os.environ.get("GAPFILLER_HOME",
                          os.path.join(os.path.expanduser("~"), ".cache", "gapfiller"))


def ensure_weights(path=None):
    if path and os.path.exists(path):
        return path
    d = default_ckpt_dir()
    os.makedirs(d, exist_ok=True)
    path = path or os.path.join(d, "gapfiller_v1.pt")
    if os.path.exists(path):
        return path
    last = None
    for url in WEIGHT_URLS:
        try:
            print(f"downloading weights from {url}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return path
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not fetch weights, place gapfiller_v1.pt in {d} ({last})")


def load_model(ckpt=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ensure_weights(ckpt), map_location=device)
    chans, nres = CFG[ck.get("size", "large")]
    model = GapFillerNet(chans=chans, nres=nres).to(device).eval()
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    return model, device


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = str(s["r_frame_rate"]).split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den or 1)


def read_frames(path, w, h, limit=0):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if limit:
        arr = arr[:limit]
    return [torch.from_numpy(f.copy()).permute(2, 0, 1).float().div(255) for f in arr]


def write_frames(frames, path, w, h, fps, crf=16):
    arr = (torch.stack(frames).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", path], input=arr.tobytes())


def interpolate_video(inp, outp, target_fps=None, multiplier=None, ckpt=None,
                      limit=0, progress=None, **knobs):
    """Retime to target_fps, or multiply frame count by multiplier."""
    model, device = load_model(ckpt)
    w, h, src = probe(inp)
    frames = read_frames(inp, w, h, limit)
    n = len(frames)
    if n < 2:
        raise ValueError("need at least two frames")

    if target_fps:
        out_fps = float(target_fps)
        nout = max(1, int(round((n - 1) / src * out_fps)) + 1)
        times = [((i / out_fps) * src) for i in range(nout)]
    else:
        m = int(multiplier or 2)
        out_fps = src * m
        times = [i / m for i in range((n - 1) * m + 1)]

    out, made, t0 = [], 0, time.time()
    for idx, ts in enumerate(times):
        j = int(math.floor(ts))
        t = ts - j
        if j >= n - 1:
            out.append(frames[-1])
        elif t < 1e-4:
            out.append(frames[j])                    # real frame, nothing to synthesise
        else:
            a, b = frames[j][None].to(device), frames[j + 1][None].to(device)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                                 enabled=(device == "cuda")):
                y = model(a, b, t=t, **knobs)
            out.append(y[0].float().clamp(0, 1).cpu())
            made += 1
        if progress and idx % 10 == 0:
            progress(idx / len(times))
    write_frames(out, outp, w, h, out_fps)
    return dict(frames_in=n, frames_out=len(out), synthesised=made,
                src_fps=src, out_fps=out_fps, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(prog="gapfiller",
                                 description="Fills the gaps between your frames.")
    ap.add_argument("input")
    ap.add_argument("output")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fps", type=float, help="retime to this frame rate (e.g. 60)")
    g.add_argument("--multiplier", type=int, help="multiply frame count (e.g. 2, 4)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N frames")
    ap.add_argument("--sharpness", type=float, default=1.0)
    ap.add_argument("--blend-bias", type=float, default=0.0)
    ap.add_argument("--flow-scale", type=float, default=1.0)
    ap.add_argument("--scene-thresh", type=float, default=0.0)
    ap.add_argument("--scale-factor", type=float, default=0.0, help="0 = auto")
    ap.add_argument("--no-ensemble", action="store_true")
    a = ap.parse_args()
    if not a.fps and not a.multiplier:
        a.fps = 60.0
    res = interpolate_video(
        a.input, a.output, a.fps, a.multiplier, a.ckpt, a.limit,
        sharpness=a.sharpness, blend_bias=a.blend_bias, flow_scale=a.flow_scale,
        scene_thresh=a.scene_thresh, scale_factor=a.scale_factor,
        ensemble=not a.no_ensemble)
    print(f"{res['frames_in']} -> {res['frames_out']} frames "
          f"({res['src_fps']:.2f} -> {res['out_fps']:.2f} fps, "
          f"{res['synthesised']} synthesised in {res['seconds']}s)")
    print(f"saved {a.output}")


def ui():
    """Drag-and-drop web UI."""
    try:
        import gradio as gr
    except ImportError:
        print("the UI needs gradio:  pip install gradio")
        sys.exit(1)
    import tempfile

    def run(video, mode, fps, mult, sharpness, ensemble, scale_factor, progress=gr.Progress()):
        if not video:
            return None, "load a video first"
        outp = os.path.join(tempfile.mkdtemp(), "gapfiller.mp4")
        res = interpolate_video(
            video, outp,
            target_fps=fps if mode == "retime to fps" else None,
            multiplier=int(mult) if mode == "multiply frames" else None,
            progress=lambda f: progress(f),
            sharpness=sharpness, ensemble=ensemble, scale_factor=scale_factor)
        return outp, (f"{res['frames_in']} -> {res['frames_out']} frames | "
                      f"{res['src_fps']:.2f} -> {res['out_fps']:.2f} fps | "
                      f"{res['synthesised']} synthesised | {res['seconds']}s")

    with gr.Blocks(title="GapFiller") as demo:
        gr.Markdown("# GapFiller\nFills the gaps between your frames.")
        with gr.Row():
            with gr.Column():
                vid = gr.Video(label="input video")
                mode = gr.Radio(["retime to fps", "multiply frames"],
                                value="retime to fps", label="mode")
                fps = gr.Slider(1, 240, value=60, step=1, label="target fps")
                mult = gr.Slider(2, 8, value=2, step=1, label="multiplier")
                with gr.Accordion("advanced", open=False):
                    sharp = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="sharpness")
                    ens = gr.Checkbox(value=True, label="ensemble (slower, more accurate)")
                    sf = gr.Dropdown([0.0, 0.25, 0.5, 1.0], value=0.0,
                                     label="flow scale (0 = auto)")
                go = gr.Button("Fill the gaps", variant="primary")
            with gr.Column():
                out = gr.Video(label="result")
                info = gr.Textbox(label="stats", interactive=False)
        go.click(run, [vid, mode, fps, mult, sharp, ens, sf], [out, info])
    demo.launch()


if __name__ == "__main__":
    main()
