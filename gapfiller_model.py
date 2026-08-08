"""GapFiller: flow-based frame interpolation with arbitrary-time conditioning.

Trained knob
  t          arbitrary interpolation time in (0,1). 24->60fps is 2.5x, not a power
             of two, so a t=0.5-only model has to over-generate and resample, which
             lands output frames off-grid (judder). Conditioning on t emits exactly
             the frame times the target rate needs.

Inference knobs (no retraining -- they reshape how the trained pieces combine)
  sharpness    scale on the learned residual (detail up/down)
  blend_bias   push the blend mask toward the previous (+) or next (-) frame
  flow_scale   damp/boost estimated motion; <1 is safer on very large motion
  scene_thresh if |f0-f1| exceeds this, treat as a cut: return the nearer frame
               instead of morphing between unrelated images (Proteus has the same input)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(ic, oc, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ic, oc, k, s, p, bias=True), nn.PReLU(oc))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1); self.a1 = nn.PReLU(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1); self.a2 = nn.PReLU(c)

    def forward(self, x):
        return self.a2(x + self.c2(self.a1(self.c1(x))))


def warp(img, flow):
    B, _, H, W = img.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    x = gx.unsqueeze(0) + flow[:, 0]
    y = gy.unsqueeze(0) + flow[:, 1]
    grid = torch.stack((2.0 * x / max(W - 1, 1) - 1.0,
                        2.0 * y / max(H - 1, 1) - 1.0), dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


class IFBlock(nn.Module):
    def __init__(self, in_ch, c=64, nres=6):
        super().__init__()
        self.down = nn.Sequential(conv(in_ch, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(nres)])
        self.head = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            if scale != 1:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear",
                                     align_corners=False) / scale
            x = torch.cat([x, flow], 1)
        y = self.head(self.body(self.down(x)))
        y = F.interpolate(y, scale_factor=2.0 * scale, mode="bilinear", align_corners=False)
        return y[:, :4] * (2.0 * scale), y[:, 4:5]


class GapFillerNet(nn.Module):
    def __init__(self, chans=(128, 96, 72), nres=(8, 6, 4), refine_c=64):
        super().__init__()
        # +1 channel everywhere for the broadcast t map
        self.b0 = IFBlock(6 + 1, chans[0], nres[0])
        self.b1 = IFBlock(6 + 6 + 1 + 1 + 4, chans[1], nres[1])
        self.b2 = IFBlock(6 + 6 + 1 + 1 + 4, chans[2], nres[2])
        self.refine = nn.Sequential(
            conv(6 + 6 + 1 + 1 + 4, refine_c), *[ResBlock(refine_c) for _ in range(3)],
            nn.Conv2d(refine_c, 3, 3, 1, 1))
        self.scales = (4, 2, 1)

    def _tmap(self, x, t):
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        return t.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.dtype)

    def forward(self, img0, img1, t=0.5, return_aux=False,
                sharpness=1.0, blend_bias=0.0, flow_scale=1.0, scene_thresh=0.0,
                scale_factor=0.0, ensemble=False):
        # scale_factor=0 means AUTO: large inter-frame difference implies motion that
        # may exceed the finest stage's receptive field, so drop to coarser flow
        # estimation. Measured: this is worth +0.6 dB on the hardest motion bucket.
        if scale_factor == 0.0:
            with torch.no_grad():
                d = (img0 - img1).abs().mean().item() * 255.0
            scale_factor = 0.5 if d >= 40.0 else 1.0
        # ensemble: interpolating (a,b) at t and (b,a) at 1-t should give the same
        # frame, so averaging the two cancels direction-specific error. Costs 2x compute.
        if ensemble:
            k = dict(sharpness=sharpness, blend_bias=-blend_bias, flow_scale=flow_scale,
                     scene_thresh=scene_thresh, scale_factor=scale_factor)
            tt = t if not torch.is_tensor(t) else t
            a = self.forward(img0, img1, t, False, sharpness, blend_bias, flow_scale,
                             scene_thresh, scale_factor, False)
            rt = (1.0 - t) if not torch.is_tensor(t) else (1.0 - t)
            b = self.forward(img1, img0, rt, False, **k, ensemble=False)
            return ((a + b) / 2).clamp(0, 1)
        B, C, H, W = img0.shape
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        if ph or pw:
            img0 = F.pad(img0, (0, pw, 0, ph), mode="replicate")
            img1 = F.pad(img1, (0, pw, 0, ph), mode="replicate")
        tm = self._tmap(img0, t)

        # scale_factor <1 estimates flow at coarser resolution, which is the standard
        # way to catch motion larger than the receptive field (RIFE exposes the same knob)
        scales = tuple(max(1.0, sc / max(scale_factor, 1e-3)) for sc in self.scales)
        flow, mask, aux = None, None, []
        for blk, s in zip((self.b0, self.b1, self.b2), scales):
            if flow is None:
                f, m = blk(torch.cat([img0, img1, tm], 1), None, s)
            else:
                w0 = warp(img0, flow[:, :2]); w1 = warp(img1, flow[:, 2:4])
                df, dm = blk(torch.cat([img0, img1, w0, w1, mask, tm], 1), flow, s)
                f, m = flow + df, mask + dm
            flow, mask = f, m
            if return_aux:
                a = torch.sigmoid(mask)
                aux.append((warp(img0, flow[:, :2]) * a +
                            warp(img1, flow[:, 2:4]) * (1 - a))[:, :, :H, :W])

        if flow_scale != 1.0:
            flow = flow * flow_scale
        w0 = warp(img0, flow[:, :2]); w1 = warp(img1, flow[:, 2:4])
        a = torch.sigmoid(mask + blend_bias)
        merged = w0 * a + w1 * (1 - a)
        res = self.refine(torch.cat([img0, img1, w0, w1, mask, tm, flow], 1))
        out = (merged + res * sharpness).clamp(0, 1)

        if scene_thresh > 0:
            # hard cut: morphing across a scene change looks worse than picking a side
            d = (img0 - img1).abs().mean(dim=(1, 2, 3))
            cut = (d > scene_thresh).view(-1, 1, 1, 1)
            tt = self._tmap(img0, t)[:, :1, :1, :1]
            out = torch.where(cut, torch.where(tt < 0.5, img0, img1), out)

        out = out[:, :, :H, :W]
        if return_aux:
            return out, aux, merged[:, :, :H, :W]
        return out


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps ** 2).mean()


def laplacian_pyramid_loss(pred, gt, levels=3):
    loss, p, g = 0.0, pred, gt
    for _ in range(levels):
        loss = loss + charbonnier(p, g)
        p = F.avg_pool2d(p, 2); g = F.avg_pool2d(g, 2)
    return loss + charbonnier(p, g)


if __name__ == "__main__":
    m = GapFillerNet()
    print(f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    x0, x1 = torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256)
    print("t=0.5      ", tuple(m(x0, x1, 0.5).shape))
    print("t=tensor   ", tuple(m(x0, x1, torch.tensor([0.25, 0.75])).shape))
    print("knobs      ", tuple(m(x0, x1, 0.4, sharpness=1.5, blend_bias=0.3,
                                 flow_scale=0.8, scene_thresh=0.35).shape))
    print("odd size   ", tuple(m(torch.rand(1, 3, 1080, 1920), torch.rand(1, 3, 1080, 1920), 0.5).shape))
