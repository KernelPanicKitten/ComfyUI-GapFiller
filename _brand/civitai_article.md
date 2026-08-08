# GapFiller: frame interpolation that actually lands on 60fps

Most AI video comes out at 24fps and looks it. The usual fix is frame
interpolation, and the usual tool is RIFE. RIFE is good. I still ended up
training my own, because there is a specific thing it does not do well, and it
was bugging me.

## The problem: 60 is not a multiple of 24

Almost every interpolator generates the **midpoint** between two frames. Give it
frame 1 and frame 2, it invents the frame halfway between them. Run it again and
you get quarter points. That gives you 2x, 4x, 8x.

24 to 60fps is 2.5x.

There is no number of halvings that lands on 2.5. So the standard approach
over-generates (24 -> 96fps) and then throws frames away to get down to 60. The
frames that survive are not evenly spaced in time. Some are real, some are a
quarter step off. That unevenness is judder, and on smooth camera moves you can
see it.

## What I did

GapFiller is conditioned on the interpolation time. Instead of only knowing "the
middle", it takes a value `t` and renders the frame at exactly that moment. Need
a frame 37% of the way between two source frames? It makes that frame. No
over-generation, no discarding, no off-grid timing.

It is a flow-based model, same family as RIFE: it estimates motion between the
two frames, warps both of them to the target time, blends them, and adds a
learned detail pass. Warping instead of inventing pixels is why the output stays
sharp. The sharp pixels already exist in your footage, the network just has to
work out where they moved to.

I trained it self-supervised on a private corpus of real and generated video.
The trick there is that you do not need a teacher model at all: take any clip,
hide the middle frame, and the real middle frame is your training label. Free,
unlimited, perfectly accurate labels.

## Does it actually beat RIFE

Short answer: yes, on my content, by a modest margin. Here are the real numbers.

279 held-out clips the model never trained on, identical inputs for every method,
same metric code. "Blend" is just averaging the two frames, the naive baseline.

| Split | Blend | RIFE 4.7 | **GapFiller** |
| --- | --- | --- | --- |
| All samples | 23.39 | 27.74 | **27.92** |
| Generated video | 23.09 | 28.27 | **28.57** |
| Fast motion | 19.71 | 23.81 | **23.85** |
| Extreme motion | 14.93 | 19.73 | **19.98** |
| Arbitrary timing | 22.79 | 27.44 | **27.74** |

PSNR in dB, higher is better. GapFiller wins every split by +0.18 to +0.30 dB.

Now the honest part, because I would rather you trust the numbers than be
impressed by them:

- **Those margins are small.** Consistent, but small. This is not a landslide.
- **My test set is my domain.** It is generated and real video clips of the kind
  I actually make. RIFE is general-purpose, trained on Vimeo90k. On generic
  footage it may well match or beat this.
- **RIFE is far more efficient.** 5.3M parameters against my 24M. They get
  almost the same quality out of a model a quarter the size. That is better
  engineering than mine, and worth saying out loud.
- **I stole one of their ideas.** RIFE has a `scale_factor` control that
  estimates motion at lower resolution to catch very large movement. I was
  *losing* the extreme-motion split until I added the same thing. Then I won it.

The place GapFiller genuinely pulls ahead is arbitrary timing, which is the whole
reason it exists.

## Controls

Six of them, all live at inference so you can tweak without retraining:

| Control | What it does |
| --- | --- |
| `sharpness` | More or less detail in the generated frames |
| `blend_bias` | Lean toward the previous or the next frame |
| `flow_scale` | Damp the motion estimate, safer on wild movement |
| `scene_thresh` | Cut instead of morphing when the shot changes |
| `ensemble` | Run both temporal directions and average. Slower, better |
| `scale_factor` | Motion-estimation resolution, automatic by default |

The automatic mode watches how much the frame changed and drops to coarse motion
estimation when things move a lot. That is what turned the extreme-motion loss
into a win, and you get it without touching anything.

## Using it

**In ComfyUI:** install from the manager, or clone into `custom_nodes`. Two nodes,
one to multiply frames and one to retime to any fps. Example workflows are in the
repo. Weights download themselves on first run.

**Not in ComfyUI:** there is a standalone version.

```
pip install comfyui-gapfiller
gapfiller input.mp4 output.mp4 --fps 60
```

There is a drag-and-drop web UI too if you would rather not touch a terminal:

```
pip install "comfyui-gapfiller[ui]"
gapfiller-ui
```

Works at any resolution. MIT licensed. Architecture credit goes to the RIFE
authors (Huang et al., ECCV 2022), whose IFNet design this belongs to the family
of, and whose large-motion trick I borrowed.

GitHub: <REPO_URL>
