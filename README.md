# ComfyUI-GapFiller

Fills the gaps between your frames.

Frame interpolation for ComfyUI with **arbitrary-time conditioning** and direct
retiming to any target frame rate.

Most interpolators only synthesise the *midpoint* between two frames. Converting
24 fps to 60 fps is a 2.5x ratio, so a midpoint-only model has to over-generate
(24 -> 96 fps) and then throw frames away, landing output frames off the intended
timeline. GapFiller is conditioned on the interpolation time `t`, so it renders
each output frame at exactly the moment the target rate needs.

## Features

- **Arbitrary-time interpolation** - any `t`, not just 0.5
- **Retime to any fps** - 24 -> 60 directly, no integer-multiplier restriction
- **Six inference controls** - sharpness, blend bias, flow scale, scene threshold,
  ensemble, and automatic multi-scale flow
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

| Split | Blend | RIFE 4.7 | **GapFiller** |
| --- | --- | --- | --- |
| All samples | 23.39 | 27.74 | **27.92** |
| Generated video | 23.09 | 28.27 | **28.57** |
| Motion 25+ | 19.71 | 23.81 | **23.85** |
| Motion 50+ (extreme) | 14.93 | 19.73 | **19.98** |
| Arbitrary `t` (not 0.5) | 22.79 | 27.44 | **27.74** |

PSNR in dB, higher is better. GapFiller leads in every split, by **+0.18 to
+0.30 dB**.

### Honest caveats

- These margins are consistent but modest. This is not a landslide.
- The evaluation set is drawn from the same domain the model was trained on
  (generated and real video clips). RIFE is a general-purpose model trained on
  Vimeo90k; on generic footage it may well match or beat this model.
- GapFiller is 24M parameters against RIFE's 5.3M. RIFE is considerably more
  parameter-efficient; the advantage here comes from domain training and
  time-conditioning, not from a better architecture per parameter.
- `scale_factor` is directly inspired by RIFE's knob of the same name. Before
  adopting it this model *lost* the extreme-motion split.

## Installation

**ComfyUI Manager**: search for `GapFiller`.

**Manual**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/KernelPanicKitten/ComfyUI-GapFiller
```

Weights download automatically on first run to `ComfyUI/models/gapfiller/`.
To install them by hand, place the `.pt` file in that directory.

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
