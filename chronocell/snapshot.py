"""
Static 3D snapshots (PNG) and animations (GIF) of a chromatin fibre, with Pillow only.

A thick polyline is drawn back to front (painter's algorithm) under an orthographic projection,
with depth cueing (far segments lighter) so the fold reads as 3D in a PDF or on a slide. Used by
the PDF dossier and the 4D GIF export; the interactive views stay in Plotly.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import theme as T

GHOST_RGB = (205, 205, 199)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[k:k + 2], 16) for k in (0, 2, 4))


def colormap(stops: list[str] | list[list], values: np.ndarray) -> np.ndarray:
    """values in [0, 1] -> (n, 3) RGB via linear interpolation between colour stops."""
    if stops and isinstance(stops[0], (list, tuple)):
        pos = np.array([float(p) for p, _ in stops])
        cols = np.array([_hex(c) for _, c in stops], float)
    else:
        cols = np.array([_hex(c) for c in stops], float)
        pos = np.linspace(0, 1, len(cols))
    v = np.clip(np.nan_to_num(np.asarray(values, float), nan=0.0), 0, 1)
    return np.stack([np.interp(v, pos, cols[:, k]) for k in range(3)], axis=1)


def normalise(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(values, float)
    ok = np.isfinite(v) if valid is None else (np.asarray(valid, bool) & np.isfinite(v))
    out = np.zeros(len(v))
    if ok.any():
        lo, hi = np.nanpercentile(v[ok], [1, 99])
        out[ok] = np.clip((v[ok] - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    return out


def _rotation(elev: float, azim: float) -> np.ndarray:
    a, e = math.radians(azim), math.radians(elev)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return rx @ rz


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(coords: np.ndarray, values01: np.ndarray, stops, valid: np.ndarray | None = None, size=(900, 700),
           elev: float = 22, azim: float = 38, title: str = "", scale_nm: float | None = None,
           bounds: tuple[np.ndarray, float] | None = None, background: str = T.PAPER) -> Image.Image:
    """PIL image of the fibre. `bounds` = (centre, half-extent) fixes the framing across frames."""
    x = np.asarray(coords, float)
    n = len(x)
    w, h = size
    img = Image.new("RGB", size, _hex(background))
    if n < 2:
        return img
    rot = _rotation(elev, azim)
    centre, half = bounds if bounds is not None else (x.mean(axis=0), float(np.max(np.linalg.norm(x - x.mean(axis=0), axis=1))))
    p = (x - centre) @ rot.T
    s = 0.46 * min(w, h) / max(half, 1e-9)
    px = w / 2 + p[:, 0] * s
    py = h / 2 - p[:, 2] * s
    depth = p[:, 1]
    rgb = colormap(stops, values01)
    if valid is not None:
        rgb[~np.asarray(valid, bool)] = GHOST_RGB
    width = int(np.clip(900 / math.sqrt(n), 2, 11) * min(w, h) / 700)
    seg_depth = (depth[:-1] + depth[1:]) / 2
    dmin, dmax = float(seg_depth.min()), float(seg_depth.max())
    order = np.argsort(-seg_depth)                       # far first
    draw = ImageDraw.Draw(img)
    paper = np.array(_hex(background), float)
    for k in order:
        f = (seg_depth[k] - dmin) / (dmax - dmin) if dmax > dmin else 0.0   # 1 = far
        col = (rgb[k] + rgb[k + 1]) / 2
        col = col * (1 - 0.45 * f) + paper * 0.45 * f
        shade = tuple(int(c) for c in col)
        a, b = (px[k], py[k]), (px[k + 1], py[k + 1])
        draw.line([a, b], fill=shade, width=width)
        r = width / 2
        draw.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], fill=shade)
    if title:
        draw.text((18, 14), title, fill=_hex(T.INK), font=_font(max(14, h // 32)))
    if scale_nm:
        length = scale_nm * s
        y0 = h - 26
        draw.line([(18, y0), (18 + length, y0)], fill=_hex(T.INK), width=3)
        draw.text((18, y0 - 22), f"{scale_nm:,.0f} nm", fill=_hex(T.INK_2), font=_font(max(12, h // 45)))
    return img


def nice_scale(coords: np.ndarray) -> float:
    """Largest 1-2-5 x 10^k length not longer than a quarter of the structure's extent."""
    ext = float(np.max(np.ptp(np.asarray(coords, float), axis=0))) if len(coords) > 1 else 1.0
    target = max(ext / 4, 1.0)
    k = 10 ** np.floor(np.log10(target))
    return float(max(m * k for m in (1, 2, 5) if m * k <= target))


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def side_by_side(left: Image.Image, right: Image.Image, gap: int = 12, background: str = T.PAPER) -> Image.Image:
    out = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), _hex(background))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


def gif(frames: np.ndarray, values01: np.ndarray, stops, labels: list[str], size=(640, 520), elev: float = 22,
        azim: float = 38, spin: float = 0.0, frame_ms: int = 140, valid: np.ndarray | None = None) -> bytes:
    """Animated GIF of T frames (fixed framing; optional slow spin in degrees per frame)."""
    allp = frames.reshape(-1, 3)
    centre = allp.mean(axis=0)
    half = float(np.max(np.linalg.norm(allp - centre, axis=1)))
    vals = values01 if values01.ndim == 2 else np.broadcast_to(values01, (len(frames), frames.shape[1]))
    imgs = [render(fx, vals[t], stops, valid if valid is not None and len(valid) == frames.shape[1] else None, size,
                   elev, azim + spin * t, labels[t] if t < len(labels) else "", bounds=(centre, half))
            .convert("P", palette=Image.ADAPTIVE, colors=128) for t, fx in enumerate(frames)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=frame_ms, loop=0, optimize=False)
    return buf.getvalue()
