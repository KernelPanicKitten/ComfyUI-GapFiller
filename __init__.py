"""GapFiller - frame interpolation nodes for ComfyUI.

Fills the gaps between your frames.

Two nodes:
  GapFiller (frame interpolate)  multiply frame count by N
  GapFiller (retime to fps)      resample a sequence to an exact target fps,
                                 generating every frame at its true time

Weights download automatically on first use into ComfyUI/models/gapfiller/.
"""
import os
import math
import glob
import urllib.request

import torch

try:
    import folder_paths
    CKPT_DIR = os.path.join(folder_paths.models_dir, "gapfiller")
except Exception:
    CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

from .gapfiller_model import GapFillerNet

DEFAULT_CKPT = "gapfiller_v1.pt"
WEIGHT_URLS = [
    "https://huggingface.co/KernelPanicKitten/GapFiller/resolve/main/gapfiller_v1.pt",
]

CFG = {
    "current": ((128, 96, 72), (8, 6, 4)),
    "bigger": ((192, 144, 96), (10, 8, 6)),
    "large": ((256, 192, 128), (12, 10, 8)),
}
_CACHE = {}


def _ensure_weights(name):
    """Download the default checkpoint on first use, like other VFI packs do."""
    path = os.path.join(CKPT_DIR, name)
    if os.path.exists(path):
        return path
    if name != DEFAULT_CKPT:
        raise FileNotFoundError(f"{path} not found")
    last = None
    for url in WEIGHT_URLS:
        try:
            print(f"[GapFiller] downloading weights from {url}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            print(f"[GapFiller] saved {path}")
            return path
        except Exception as exc:  # try the next mirror
            last = exc
    raise RuntimeError(
        f"could not download GapFiller weights; place {name} in {CKPT_DIR} manually ({last})"
    )


def _load(name, device):
    key = (name, str(device))
    if key not in _CACHE:
        path = _ensure_weights(name)
        ck = torch.load(path, map_location=device)
        chans, nres = CFG[ck.get("size", "large")]
        model = GapFillerNet(chans=chans, nres=nres).to(device).eval()
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        _CACHE[key] = model
    return _CACHE[key]


def _ckpts():
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    return found or [DEFAULT_CKPT]


KNOBS = {
    "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05,
                            "tooltip": "Scales the learned detail. Above 1 is crisper, below 1 softer."}),
    "blend_bias": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                             "tooltip": "Biases the blend toward the previous (+) or next (-) frame."}),
    "flow_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                             "tooltip": "Damps estimated motion. Below 1 is safer on very large motion."}),
    "scene_thresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "Above this frame difference, cut instead of morphing. 0 disables."}),
    "ensemble": ("BOOLEAN", {"default": True,
                             "tooltip": "Average both temporal directions. Roughly 2x compute, more accurate."}),
    "scale_factor": ([0.0, 0.25, 0.5, 1.0], {"default": 0.0,
                     "tooltip": "Flow resolution. 0 is automatic (coarse flow on large motion)."}),
}


def _run(model, a, b, t, device, knobs):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(a[None].to(device), b[None].to(device), t=t, **knobs)
    return out[0].float().clamp(0, 1).cpu()


class GapFillerInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
                "ckpt_name": (_ckpts(),),
                **KNOBS,
            },
            "optional": {
                "loop": ("BOOLEAN", {"default": False,
                                     "tooltip": "Also interpolate last to first for a seamless loop."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "GapFiller"

    @torch.no_grad()
    def interpolate(self, images, multiplier, ckpt_name, sharpness, blend_bias,
                    flow_scale, scene_thresh, ensemble, scale_factor, loop=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load(ckpt_name, device)
        knobs = dict(sharpness=sharpness, blend_bias=blend_bias, flow_scale=flow_scale,
                     scene_thresh=scene_thresh, ensemble=ensemble, scale_factor=scale_factor)
        seq = [images[i].permute(2, 0, 1).contiguous() for i in range(images.shape[0])]
        if loop and len(seq) > 1:
            seq = seq + [seq[0]]
        if len(seq) < 2:
            return (images,)
        out = []
        for i in range(len(seq) - 1):
            out.append(seq[i])
            for j in range(1, multiplier):
                out.append(_run(model, seq[i], seq[i + 1], j / multiplier, device, knobs))
        if not loop:
            out.append(seq[-1])
        return (torch.stack([f.permute(1, 2, 0) for f in out]).clamp(0, 1),)


class GapFillerRetime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "src_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "target_fps": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "ckpt_name": (_ckpts(),),
                **KNOBS,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "retime"
    CATEGORY = "GapFiller"

    @torch.no_grad()
    def retime(self, images, src_fps, target_fps, ckpt_name, sharpness, blend_bias,
               flow_scale, scene_thresh, ensemble, scale_factor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load(ckpt_name, device)
        knobs = dict(sharpness=sharpness, blend_bias=blend_bias, flow_scale=flow_scale,
                     scene_thresh=scene_thresh, ensemble=ensemble, scale_factor=scale_factor)
        seq = [images[i].permute(2, 0, 1).contiguous() for i in range(images.shape[0])]
        n = len(seq)
        if n < 2:
            return (images,)
        nout = max(1, int(round((n - 1) / src_fps * target_fps)) + 1)
        out = []
        for i in range(nout):
            ts = (i / target_fps) * src_fps
            j = int(math.floor(ts))
            t = ts - j
            if j >= n - 1:
                out.append(seq[-1])
            elif t < 1e-4:
                out.append(seq[j])          # lands on a real frame, no synthesis needed
            else:
                out.append(_run(model, seq[j], seq[j + 1], t, device, knobs))
        return (torch.stack([f.permute(1, 2, 0) for f in out]).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {
    "GapFillerInterpolate": GapFillerInterpolate,
    "GapFillerRetime": GapFillerRetime,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GapFillerInterpolate": "GapFiller (frame interpolate)",
    "GapFillerRetime": "GapFiller (retime to fps)",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
