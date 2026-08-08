# ComfyUI-GapFiller

Fills the gaps between your frames.

> **Early preview (v0.1).** First public release, trained in a few hours on a
> single GPU. Quality is on par with RIFE rather than better than it (see the
> benchmark section, which says so plainly). A retrain is planned. Failure cases
> are genuinely welcome.

Frame interpolation for ComfyUI with **arbitrary-time conditioning** and direct
retiming to any target frame rate.

Frame-rate conversions rarely land on whole numbers. 24 -> 60 fps is 2.5x; 16 -> 60
fps (common with Wan workflows) is 3.75x. The ComfyUI RIFE node only multiplies by
whole numbers, so reaching 60 means picking a multiple that overshoots and then
dropping frames yourself: 5x to 120 fps and bin half, or 4x to 64 fps and resample
unevenly down to 60.

GapFiller is conditioned on the interpolation time `t`, so it renders each output
frame at exactly the moment the target rate needs. Tell it the source and target
fps and it generates precisely those frames: no over-generation, no discard pass,
no arithmetic, one node.

## Features

- **Arbitrary-time interpolation** - any `t`, not just 0.5
- **Retime to any fps** - 16 -> 60 or 24 -> 60 directly, no integer-multiplier
  restriction and no manual frame dropping
- **Six inference controls** - sharpness, blend bias, flow scale, scene threshold,
  ensemble and flow resolution. Four of these have no equivalent in the ComfyUI
  RIFE node, which exposes `fast_mode`, `ensemble` and `scale_factor`.
- **Any resolution** - fully convolutional, pads internally
- **Automatic weight download** on first use

## Nodes

| Node | Purpose |
| --- | --- |
| **GapFiller (frame interpolate)** | Insert `multiplier - 1` frames between every input pair |
| **GapFiller (retime to fps)** | Resample a sequence from `src_fps` to `target_fps` |

### Controls

| Control | Default | Effect |
| --- | --- | --- |
| `sharpness` | 1.0 | Scales the learned detail residual. Lower is softer. |
| `blend_bias` | 0.0 | Biases the blend toward the previous (+) or next (-) frame. |
| `flow_scale` | 1.0 | Damps estimated motion. Below 1.0 is safer on very large motion. |
| `scene_thresh` | 0.0 | Above this frame difference, cut instead of morphing. 0 disables. |
| `ensemble` | true | Averages both temporal directions. Roughly 2x compute, more accurate. |
| `scale_factor` | 0 (auto) | Flow-estimation resolution. Auto drops to coarse flow on large motion. |

## Benchmark

Measured on 279 held-out samples that the model never trained on, with identical
inputs and metric code for every method. `blend` is the naive average of the two
input frames.

Compared against every RIFE checkpoint the ComfyUI node supports, including 4.26,
their newest release. PSNR in dB, higher is better.

| Split | Blend | RIFE 4.7 | RIFE 4.17 | RIFE 4.26 | GapFiller |
| --- | --- | --- | --- | --- | --- |
| All samples | 23.39 | 27.74 | 27.82 | 27.73 | 27.88 |
| Generated video | 23.09 | 28.27 | 28.53 | 28.38 | 28.52 |
| Real footage | 23.69 | 27.20 | 27.09 | 27.06 | 27.22 |
| Midpoint frames | 23.78 | 27.93 | 27.88 | - | 27.99 |

### Read this before quoting the table

**This is a tie, not a win.** Every gap between GapFiller and current RIFE is
0.01 to 0.13 dB, which is noise. GapFiller is marginally ahead on some splits and
marginally behind on others.

- **RIFE is far more parameter-efficient.** 5.4M parameters against GapFiller's
  24M for the same quality. That is better engineering.
- **The evaluation set matches the training domain** (generated and real video
  clips). RIFE is a generalist trained on Vimeo90k; on general footage it may
  well come out ahead.
- **PSNR may flatter this model.** The RIFE authors state in their repository
  that "improving the PSNR index is not consistent with subjective perception"
  and tune for perceptual quality instead of the metric.
- **RIFE ships versions the ComfyUI node does not expose** (4.18 through 4.25).
  Those are untested here.
- **`scale_factor` is RIFE's idea**, adopted here. Without it this model *lost*
  the large-motion split.

The reason to use GapFiller is the controls and the direct fps retiming, not a
quality advantage.

## Installation

Search for **GapFiller** in ComfyUI Manager and install it, or from the command
line:

```bash
comfy node install comfyui-gapfiller
```

To install from source instead:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/KernelPanicKitten/ComfyUI-GapFiller
```

Either way, restart ComfyUI afterwards. The nodes appear under the **GapFiller**
category.

Weights (96 MB) download automatically the first time a node runs, into
`ComfyUI/models/gapfiller/`. To place them by hand instead, download
`gapfiller_v1.pt` from the
[releases page](https://github.com/KernelPanicKitten/ComfyUI-GapFiller/releases)
into that directory.

## Example workflows

Drag either file from `example_workflows/` onto the ComfyUI canvas:

| Workflow | What it does |
| --- | --- |
| `gapfiller_retime_24_to_60.json` | Retimes a 24 fps clip to 60 fps |
| `gapfiller_2x_interpolate.json` | Doubles the frame count |

Both use [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
to load and save video.

## Without ComfyUI

GapFiller also runs standalone.

```bash
pip install comfyui-gapfiller

# retime to 60 fps
gapfiller input.mp4 output.mp4 --fps 60

# or just double the frames
gapfiller input.mp4 output.mp4 --multiplier 2
```

Drag-and-drop web UI:

```bash
pip install "comfyui-gapfiller[ui]"
gapfiller-ui
```

Weights download on first run to `~/.cache/gapfiller/`. ffmpeg must be on PATH.

## Requirements

PyTorch 2.0 or newer. No other dependencies beyond what ComfyUI already ships.

## Model

A flow-based interpolator in the IFNet family: three coarse-to-fine stages
estimate bidirectional optical flow and a blend mask, both input frames are
backward-warped to time `t`, blended, and a learned residual adds detail.
Warping rather than regressing pixels is what keeps output sharp, since the
sharp pixels already exist in the inputs.

Trained self-supervised: the real middle frame is the label, so no teacher model
is required. Training used a private corpus of real and generated video.

## Credits

The architecture belongs to the IFNet family introduced by
[RIFE](https://github.com/hzwer/ECCV2022-RIFE) (Huang et al., ECCV 2022). This is
an independent implementation, and the `scale_factor` control follows RIFE's
approach to large motion.

## License

MIT
