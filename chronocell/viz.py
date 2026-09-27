"""
Figure builders. Pure functions of (data, display options) -> plotly Figure; no physics here.

The 3D backbone is a lit triangle mesh (a real tube, not a fixed-pixel line): a Catmull-Rom
spline through the beads, swept by a circle whose frame is made rotation-minimising with a
vectorised cumulative-twist correction, so adjacent rings never shear or pinch. Level of
detail (spline density x ring sides) adapts to the window so a full chromosome stays under
~70k vertices.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import genome, theme as T

# ----------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------
def catmull_rom(points: np.ndarray, subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform Catmull-Rom spline through every bead. Returns (samples, fractional bead index)."""
    n = len(points)
    if n < 4 or subdiv <= 1:
        return points.copy(), np.arange(n, dtype=np.float64)
    p = np.vstack([points[:1], points, points[-1:]])
    p0, p1, p2, p3 = p[:-3], p[1:-2], p[2:-1], p[3:]
    t = np.linspace(0.0, 1.0, subdiv, endpoint=False)[None, :, None]
    seg = 0.5 * (2 * p1[:, None] + (p2 - p0)[:, None] * t
                 + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None] * t ** 2
                 + (3 * p1 - p0 - 3 * p2 + p3)[:, None] * t ** 3)
    samples = np.vstack([seg.reshape(-1, 3), points[-1:]])
    param = np.concatenate([(np.arange(n - 1)[:, None] + t[0, :, 0][None, :]).ravel(), [n - 1.0]])
    return samples, param


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def tube_mesh(path: np.ndarray, radius: float, sides: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (M*sides + 2, 3) and triangles (F, 3) of a capped tube along `path`."""
    m = len(path)
    tan = _unit(np.gradient(path, axis=0))
    ref = np.where(np.abs(tan[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    n0 = _unit(np.cross(tan, ref))
    b0 = np.cross(tan, n0)
    # Transport ring i-1's normal into ring i's plane; the angle it makes with n0_i is the twist
    # to undo. Accumulating those angles gives a rotation-minimising frame (to O(dtheta^2)).
    q = _unit(n0[:-1] - np.sum(n0[:-1] * tan[1:], axis=1, keepdims=True) * tan[1:])
    phi = np.arctan2(np.sum(q * b0[1:], axis=1), np.sum(q * n0[1:], axis=1))
    psi = np.concatenate([[0.0], np.cumsum(phi)])[:, None]
    nrm = np.cos(psi) * n0 + np.sin(psi) * b0
    bin_ = np.cross(tan, nrm)
    ang = 2 * np.pi * np.arange(sides) / sides
    ring = radius * (np.cos(ang)[None, :, None] * nrm[:, None, :] + np.sin(ang)[None, :, None] * bin_[:, None, :])
    verts = (path[:, None, :] + ring).reshape(-1, 3)
    verts = np.vstack([verts, path[:1], path[-1:]])                    # cap centres

    r = np.arange(m - 1)[:, None]
    s = np.arange(sides)[None, :]
    a = r * sides + s
    b = r * sides + (s + 1) % sides
    c = a + sides
    d = b + sides
    body = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    c0, c1 = m * sides, m * sides + 1
    s1 = np.arange(sides)
    cap0 = np.stack([np.full(sides, c0), (s1 + 1) % sides, s1], -1)
    cap1 = np.stack([np.full(sides, c1), (m - 1) * sides + s1, (m - 1) * sides + (s1 + 1) % sides], -1)
    return verts.astype(np.float32), np.concatenate([body, cap0, cap1]).astype(np.int32)


def level_of_detail(n_beads: int) -> tuple[int, int]:
    """(spline samples per bead, ring sides) keeping the mesh below ~70k vertices."""
    if n_beads <= 700:
        return 5, 12
    if n_beads <= 1600:
        return 4, 10
    if n_beads <= 3200:
        return 3, 8
    return 2, 6


# ----------------------------------------------------------------------------------------
# Colour encoding: data value + two flat states (unassembled, outside highlight)
# ----------------------------------------------------------------------------------------
_GHOST_MAX, _CONTEXT_MAX, _DATA_MIN = 0.015, 0.045, 0.06


def state_colorscale(scale: str, focus_color: str | None = None) -> list[list]:
    stops = T.config_stops(scale) if focus_color is None else [[0.0, focus_color], [1.0, focus_color]]
    body = [[_DATA_MIN + (1 - _DATA_MIN) * p, c] for p, c in stops]
    return [[0.0, T.GHOST], [_GHOST_MAX, T.GHOST], [_GHOST_MAX, T.CONTEXT], [_CONTEXT_MAX, T.CONTEXT]] + body


def encode(values: np.ndarray, valid: np.ndarray, focus: np.ndarray | None) -> np.ndarray:
    """Per-bead intensity in [0, 1]: 0 = unassembled, 0.03 = outside focus, >= 0.06 = data."""
    v = np.asarray(values, dtype=np.float64)
    ok = valid & np.isfinite(v)
    out = np.zeros(len(v))
    if ok.any():
        lo, hi = np.nanpercentile(v[ok], [1, 99])
        span = hi - lo if hi > lo else 1.0
        out[ok] = _DATA_MIN + (1 - _DATA_MIN) * np.clip((v[ok] - lo) / span, 0, 1)
    if focus is not None:
        out[ok & ~focus] = 0.03
    return out


def _ring_values(intensity: np.ndarray, param: np.ndarray) -> np.ndarray:
    """Interpolate data values along the spline, but snap to the nearest bead next to a flat
    state so 'unassembled' / 'outside highlight' never blend into data colours."""
    n = len(intensity)
    lo = np.clip(np.floor(param).astype(int), 0, n - 1)
    hi = np.minimum(lo + 1, n - 1)
    near_state = (intensity[lo] < _DATA_MIN) | (intensity[hi] < _DATA_MIN)
    nearest = intensity[np.clip(np.rint(param).astype(int), 0, n - 1)]
    return np.where(near_state, nearest, np.interp(param, np.arange(n), intensity))


# ----------------------------------------------------------------------------------------
# 3D viewport
# ----------------------------------------------------------------------------------------
def _camera(eye: tuple[float, float, float]) -> dict:
    return dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))


CAMERAS = {"Iso": (1.55, 1.2, 0.75), "Front": (0.0, 2.1, 0.0), "Top": (0.0, 0.01, 2.1), "Side": (2.1, 0.0, 0.0)}


def _tips(chrom_name: str) -> tuple[str, str]:
    locus = f"<b>{chrom_name}:" + "%{customdata[1]:,.0f}–%{customdata[2]:,.0f}</b><br>bin %{customdata[0]:,.0f}<br>"
    return (locus + "GC %{customdata[3]:.3f} · H3K27ac %{customdata[4]:.2f}<extra></extra>",
            locus + "unassembled (N) · no sequence<extra></extra>")


def _submesh(verts: np.ndarray, faces: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faces selected by `keep`, with vertices compacted and re-indexed (no duplicated payload)."""
    f = faces[keep]
    used = np.unique(f)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return used, remap[f].astype(np.int32), f


def viewport(sub: np.ndarray, idx: np.ndarray, intensity: np.ndarray, hover: list[str], *, scale: str,
             focus_color: str | None, style: str, radius: float, bead_px: int, height: int,
             context: np.ndarray | None, uirevision: str, scale_bar_nm: float,
             gc: np.ndarray, epi: np.ndarray, valid: np.ndarray, chrom: genome.Chrom | None = None) -> go.Figure:
    chrom = chrom or genome.DEFAULT
    fig = go.Figure()
    n = len(sub)
    cs = state_colorscale(scale, focus_color)

    if context is not None:
        fig.add_trace(go.Scatter3d(x=context[:, 0], y=context[:, 1], z=context[:, 2], mode="lines",
                                   line=dict(color="rgba(150,150,143,0.35)", width=2), hoverinfo="skip",
                                   showlegend=False))

    if style == "Tube":
        subdiv, sides = level_of_detail(n)
        path, param = catmull_rom(sub, subdiv)
        verts, faces = tube_mesh(path, radius, sides)
        ring_val = _ring_values(intensity, param)
        vint = np.concatenate([np.repeat(ring_val, sides), ring_val[:1], ring_val[-1:]]).astype(np.float32)
        ring_bead = np.clip(np.rint(param).astype(np.int64), 0, n - 1)
        vbead = np.concatenate([np.repeat(ring_bead, sides), [0, n - 1]])
        gb = idx[vbead]
        custom = np.column_stack([gb, chrom.bin_start(gb) + 1, chrom.bin_end(gb),
                                  np.nan_to_num(gc[vbead], nan=0.0), np.nan_to_num(epi[vbead], nan=0.0)])
        # float64 on purpose: float32 is exact only to 2^24 = 16.7 Mb, so chr22 loci would round.
        on_data = valid[vbead][faces].all(axis=1)       # a face is 'unassembled' if any corner is
        tip_data, tip_gap = _tips(chrom.name)
        for keep, tip in ((on_data, tip_data), (~on_data, tip_gap)):
            if not keep.any():
                continue
            used, f, _ = _submesh(verts, faces, keep)
            fig.add_trace(go.Mesh3d(
                x=verts[used, 0], y=verts[used, 1], z=verts[used, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                intensity=vint[used], intensitymode="vertex", colorscale=cs, cmin=0, cmax=1, showscale=False,
                flatshading=False, customdata=custom[used], hovertemplate=tip,
                lighting=dict(ambient=0.42, diffuse=0.78, specular=0.28, roughness=0.5, fresnel=0.12),
                lightposition=dict(x=1600, y=1200, z=2400)))
    elif style == "Line":
        path, param = catmull_rom(sub, 3)
        fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", hoverinfo="skip",
                                   line=dict(width=max(2, bead_px), color=_ring_values(intensity, param),
                                             colorscale=cs, cmin=0, cmax=1)))

    # Beads carry the locus tooltip in Beads / Line styles (the tube mesh carries its own).
    if style != "Tube":
        beads_visible = style == "Beads"
        fig.add_trace(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2], mode="markers",
            marker=dict(size=bead_px if beads_visible else 3, color=intensity, colorscale=cs, cmin=0, cmax=1,
                        opacity=1.0 if beads_visible else 0.01, line=dict(width=0)),
            text=hover, hovertemplate="%{text}<extra></extra>", showlegend=False))

    # 5' / 3' ends and a physical scale bar (units are real nanometres)
    lo_c, hi_c = sub.min(axis=0), sub.max(axis=0)
    fig.add_trace(go.Scatter3d(
        x=[sub[0, 0], sub[-1, 0]], y=[sub[0, 1], sub[-1, 1]], z=[sub[0, 2], sub[-1, 2]], mode="markers+text",
        marker=dict(size=5, color=T.INK, symbol="circle"), text=[f"  {idx[0]}", f"  {idx[-1]}"],
        textfont=dict(family=T.MONO, size=11, color=T.INK_2), textposition="middle right", hoverinfo="skip",
        showlegend=False))
    bar0 = np.array([lo_c[0], lo_c[1], lo_c[2]])
    fig.add_trace(go.Scatter3d(
        x=[bar0[0], bar0[0] + scale_bar_nm], y=[bar0[1]] * 2, z=[bar0[2]] * 2, mode="lines+text",
        line=dict(color=T.INK, width=3), text=["", f"{scale_bar_nm:,.0f} nm"], textposition="top left",
        textfont=dict(family=T.SANS, size=11, color=T.INK_2), hoverinfo="skip", showlegend=False))

    hidden = dict(visible=False, showbackground=False, showspikes=False)
    r_eye = float(np.linalg.norm(CAMERAS["Iso"][:2]))
    frames = [go.Frame(name=f"t{k}", layout=dict(scene=dict(camera=_camera(
        (r_eye * np.cos(a), r_eye * np.sin(a), CAMERAS["Iso"][2])))))
        for k, a in enumerate(np.linspace(np.arctan2(1.2, 1.55), np.arctan2(1.2, 1.55) + 2 * np.pi, 73)[1:])]
    fig.frames = frames
    btn = dict(bgcolor=T.PAPER_RAISED, bordercolor=T.RULE, borderwidth=1, font=dict(family=T.SANS, size=12, color=T.INK),
               showactive=False, type="buttons", direction="right", pad=dict(l=0, r=6, t=0, b=0))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        uirevision=uirevision, showlegend=False,
        hoverlabel=dict(bgcolor=T.PAPER_RAISED, bordercolor=T.RULE_STRONG, align="left",
                        font=dict(family=T.MONO, size=12, color=T.INK)),
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="data", bgcolor="rgba(0,0,0,0)",
                   camera=_camera(CAMERAS["Iso"]), dragmode="turntable"),
        updatemenus=[
            dict(btn, x=0.0, y=0.0, xanchor="left", yanchor="bottom", buttons=[
                dict(label="Turntable", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True), transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]),
            dict(btn, x=1.0, y=0.0, xanchor="right", yanchor="bottom", buttons=[
                dict(label=name, method="relayout", args=[{"scene.camera": _camera(eye)}])
                for name, eye in CAMERAS.items()]),
        ],
    )
    return fig


# ----------------------------------------------------------------------------------------
# 2D analytics
# ----------------------------------------------------------------------------------------
def scaling_chart(s: np.ndarray, r: np.ndarray, fit_range: tuple[int, int], nu: float, height: int = 250,
                  resolution: int = genome.RESOLUTION) -> go.Figure:
    kb = s * resolution / 1000
    fig = go.Figure()
    sel = (s >= fit_range[0]) & (s <= fit_range[1])
    if sel.sum() >= 2:
        a = int(np.flatnonzero(sel)[0])
        for nu_ref, label, dash in ((1 / 3, "1/3", "dot"), (0.5, "1/2", "dash"), (0.5876, "0.59", "dashdot")):
            ref = r[a] * (s / s[a]) ** nu_ref
            fig.add_trace(go.Scatter(x=kb, y=ref, mode="lines", line=dict(color=T.RULE_STRONG, width=1, dash=dash),
                                     hoverinfo="skip"))
            fig.add_annotation(x=np.log10(kb[-1]), y=np.log10(ref[-1]), text=f"ν={label}", showarrow=False,
                               xanchor="left", font=dict(size=10, color=T.MUTED))
    fig.add_trace(go.Scatter(x=kb, y=r, mode="lines+markers", line=dict(color=T.INK, width=1.5),
                             marker=dict(size=4, color=T.INK),
                             hovertemplate="s = %{x:,.0f} kb<br>√⟨R²⟩ = %{y:,.0f} nm<extra></extra>"))
    if sel.any() and np.isfinite(nu):
        fig.add_trace(go.Scatter(x=kb[sel], y=r[sel], mode="lines", line=dict(color=T.ACCENT, width=3), hoverinfo="skip"))
    fig.update_layout(**T.plot_layout(height, margin=dict(l=56, r=44, t=6, b=40),
                                      xaxis=dict(type="log", title="Genomic separation s (kb)"),
                                      yaxis=dict(type="log", title="RMS distance (nm)")))
    return fig


def bond_histogram(bonds_b0: np.ndarray, height: int = 170) -> go.Figure:
    fig = go.Figure(go.Histogram(x=bonds_b0, nbinsx=50, marker=dict(color=T.INK_2, line=dict(width=0)),
                                 hovertemplate="%{x} b₀<br>%{y} bonds<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=T.ACCENT, width=1.5, dash="dash"))
    fig.update_layout(**T.plot_layout(height, bargap=0.05, xaxis=dict(title="Bond length / b₀", tickformat=".2f"),
                                      yaxis=dict(title="Bonds")))
    return fig


def tracks_chart(idx: np.ndarray, gc: np.ndarray, epi: np.ndarray, focus_runs: list[tuple[int, int]],
                 chrom: genome.Chrom | None = None,
                 height: int = 250) -> go.Figure:
    chrom = chrom or genome.DEFAULT
    mb = chrom.bin_start(idx) / 1e6
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=mb, y=gc, mode="lines", line=dict(color=T.ACCENT, width=1.2), connectgaps=False,
                             hovertemplate="%{x:.2f} Mb<br>f_GC %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=mb, y=epi, mode="lines", line=dict(color=T.TERRACOTTA, width=1), fill="tozeroy",
                             fillcolor="rgba(194,74,30,0.16)", connectgaps=False,
                             hovertemplate="%{x:.2f} Mb<br>H3K27ac %{y:.2f}<extra></extra>"), row=2, col=1)
    for a, b in focus_runs:
        for r in (1, 2):
            fig.add_vrect(x0=chrom.bin_start(a) / 1e6, x1=chrom.bin_start(b) / 1e6, fillcolor=T.ACCENT_SOFT,
                          line_width=0, layer="below", row=r, col=1)
    lay = T.plot_layout(height, margin=dict(l=52, r=12, t=6, b=36))
    fig.update_layout(**{k: v for k, v in lay.items() if k not in ("xaxis", "yaxis")})
    fig.update_xaxes(**lay["xaxis"])
    fig.update_yaxes(**lay["yaxis"])
    fig.update_xaxes(title_text="chr22 position (Mb)", row=2, col=1)
    fig.update_yaxes(title_text="f_GC", row=1, col=1)
    fig.update_yaxes(title_text="H3K27ac", row=2, col=1)
    return fig


def matrix_chart(mat: np.ndarray, lo: int, k: int, kind: str, height: int = 330,
                 resolution: int = genome.RESOLUTION) -> go.Figure:
    m = mat.shape[0]
    mb = (lo + (np.arange(m) + 0.5) * k) * resolution / 1e6
    if kind == "contacts":
        z = np.log10(1 + mat)
        cs, title, fmt = T.CONTACT_SCALE, "log₁₀(1+M)", "%{z:.2f}"
    else:
        z, cs, title, fmt = mat, T.DISTANCE_SCALE, "nm", "%{z:,.0f} nm"
    fig = go.Figure(go.Heatmap(z=z, x=mb, y=mb, colorscale=cs,
                               hovertemplate="%{x:.2f} × %{y:.2f} Mb<br>" + fmt + "<extra></extra>",
                               colorbar=dict(thickness=8, len=0.8, outlinewidth=0, title=dict(text=title, side="right"),
                                             tickfont=dict(size=10, color=T.MUTED))))
    fig.update_layout(**T.plot_layout(height, margin=dict(l=48, r=8, t=6, b=40),
                                      xaxis=dict(title="Mb", constrain="domain", showgrid=False),
                                      yaxis=dict(title="Mb", autorange="reversed", scaleanchor="x", constrain="domain",
                                                 showgrid=False)))
    return fig


def decay_chart(s: np.ndarray, p: np.ndarray, gamma: float, height: int = 200,
                resolution: int = genome.RESOLUTION) -> go.Figure:
    kb = s * resolution / 1000
    ok = p > 0
    fig = go.Figure(go.Scatter(x=kb[ok], y=p[ok], mode="lines+markers", line=dict(color=T.TERRACOTTA, width=1.5),
                               marker=dict(size=4), hovertemplate="s = %{x:,.0f} kb<br>P = %{y:.3g}<extra></extra>"))
    fig.update_layout(**T.plot_layout(height, margin=dict(l=56, r=12, t=6, b=40),
                                      xaxis=dict(type="log", title="Genomic separation s (kb)"),
                                      yaxis=dict(type="log", title="Mean contacts / pair", exponentformat="power")))
    return fig


def loss_chart(history: dict[str, list], lam_smooth: float, lam_steric: float, height: int = 250) -> go.Figure:
    ep = np.asarray(history["epoch"])
    series = (("contact", 1.0, T.ACCENT, "L_contact"), ("smooth", lam_smooth, T.INK_2, "λ₁·L_smooth"),
              ("steric", lam_steric, T.TERRACOTTA, "λ₂·L_steric"))
    fig = go.Figure()
    for key, w, color, name in series:
        y = np.maximum(np.asarray(history[key]) * w, 1e-6)
        fig.add_trace(go.Scatter(x=ep, y=y, mode="lines", name=name, line=dict(color=color, width=1.4),
                                 hovertemplate=f"epoch %{{x}}<br>{name} %{{y:.4g}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ep, y=np.asarray(history["total"]), mode="lines", name="L_total",
                             line=dict(color=T.INK, width=2), hovertemplate="epoch %{x}<br>L_total %{y:.4g}<extra></extra>"))
    stages = np.asarray(history["stage"])
    if (stages == "refine").any():
        x0 = float(ep[np.argmax(stages == "refine")])
        fig.add_vline(x=x0, line=dict(color=T.RULE_STRONG, width=1, dash="dot"))
        fig.add_annotation(x=x0, y=1, yref="paper", text=" EGNN refine", showarrow=False, xanchor="left",
                           yanchor="top", font=dict(size=10, color=T.MUTED))
    fig.update_layout(**T.plot_layout(height, showlegend=True, margin=dict(l=52, r=12, t=6, b=40),
                                      legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=11)),
                                      xaxis=dict(title="Epoch"), yaxis=dict(type="log", title="Loss", exponentformat="power")))
    return fig


# ----------------------------------------------------------------------------------------
# 4D: animated trajectories and per-frame analytics
# ----------------------------------------------------------------------------------------
def trajectory_figure(frames: np.ndarray, values: np.ndarray, colorscale: list, labels: list[str], hover: list[str],
                      *, height: int = 640, bead_px: int = 3, line_px: int = 5, frame_ms: int = 120,
                      uirevision: str = "traj") -> go.Figure:
    """Client-side animation of T frames (play / pause / scrub), no server round-trip per frame.

    `values` is (M,) for a fixed colouring or (T, M) for a per-frame colouring (e.g. displacement).
    Axis ranges are fixed over all frames so the camera never jumps.
    """
    t_n = frames.shape[0]
    vals = values if values.ndim == 2 else np.broadcast_to(values, (t_n, len(values)))
    cmin, cmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if cmax <= cmin:
        cmax = cmin + 1.0

    def trace(t: int) -> go.Scatter3d:
        x = frames[t]
        return go.Scatter3d(
            x=x[:, 0], y=x[:, 1], z=x[:, 2], mode="lines+markers", text=hover,
            hovertemplate="%{text}<extra></extra>",
            line=dict(width=line_px, color=vals[t], colorscale=colorscale, cmin=cmin, cmax=cmax),
            marker=dict(size=bead_px, color=vals[t], colorscale=colorscale, cmin=cmin, cmax=cmax, line=dict(width=0)))

    fig = go.Figure(data=[trace(0)],
                    frames=[go.Frame(data=[trace(t)], traces=[0], name=str(t)) for t in range(t_n)])
    lo = frames.reshape(-1, 3).min(axis=0)
    hi = frames.reshape(-1, 3).max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    axis = lambda k: dict(visible=False, showbackground=False, range=[float(lo[k]), float(hi[k])])  # noqa: E731
    play = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        uirevision=uirevision,
        hoverlabel=dict(bgcolor=T.PAPER_RAISED, bordercolor=T.RULE_STRONG, font=dict(family=T.MONO, size=12, color=T.INK)),
        scene=dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode="manual",
                   aspectratio=dict(x=span[0] / span.max(), y=span[1] / span.max(), z=span[2] / span.max()),
                   camera=_camera(CAMERAS["Iso"]), dragmode="turntable", bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=0.0, xanchor="left", yanchor="bottom",
                          bgcolor=T.PAPER_RAISED, bordercolor=T.RULE, font=dict(family=T.SANS, size=12, color=T.INK),
                          showactive=False, pad=dict(l=0, r=6, t=0, b=0),
                          buttons=[dict(label="Play", method="animate", args=[None, play]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.16, y=0.0, len=0.84, xanchor="left", yanchor="bottom", pad=dict(t=0, b=6),
                      bgcolor=T.RULE, bordercolor=T.RULE, activebgcolor=T.ACCENT, tickcolor=T.RULE_STRONG,
                      font=dict(family=T.MONO, size=10, color=T.MUTED),
                      currentvalue=dict(prefix="", visible=True, xanchor="right",
                                        font=dict(family=T.SANS, size=12, color=T.INK_2)),
                      steps=[dict(label=labels[t], method="animate",
                                  args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                                                       transition=dict(duration=0))]) for t in range(t_n)])],
    )
    return fig


def timeseries_chart(times: np.ndarray, series: list[tuple[str, np.ndarray, str]], time_label: str,
                     height: int = 330) -> go.Figure:
    """Small multiples of per-frame metrics sharing the time axis."""
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for r, (name, y, color) in enumerate(series, start=1):
        fig.add_trace(go.Scatter(x=times, y=y, mode="lines+markers", line=dict(color=color, width=1.6),
                                 marker=dict(size=4), hovertemplate=f"%{{x}}<br>{name} %{{y:,.1f}}<extra></extra>"),
                      row=r, col=1)
        fig.update_yaxes(title_text=name, row=r, col=1)
    lay = T.plot_layout(height, margin=dict(l=64, r=12, t=6, b=40))
    fig.update_layout(**{k: v for k, v in lay.items() if k not in ("xaxis", "yaxis")})
    fig.update_xaxes(**lay["xaxis"])
    fig.update_yaxes(**lay["yaxis"])
    fig.update_xaxes(title_text=time_label, row=len(series), col=1)
    return fig


def displacement_profile(position_mb: np.ndarray, disp_nm: np.ndarray, kind: np.ndarray, height: int = 220) -> go.Figure:
    """Per-bead displacement between the first and last frame, along the (derived) chain."""
    colors = np.array([T.INK_2, T.OCHRE, T.TERRACOTTA, T.VIOLET])[np.clip(kind, 0, 3)]
    fig = go.Figure(go.Scatter(x=np.arange(len(disp_nm)), y=disp_nm, mode="markers",
                               marker=dict(size=3, color=colors), customdata=position_mb,
                               hovertemplate="bead %{x:,}<br>%{customdata:.2f} Mb<br>%{y:,.0f} nm<extra></extra>"))
    fig.update_layout(**T.plot_layout(height, margin=dict(l=56, r=12, t=6, b=40),
                                      xaxis=dict(title="Bead along the (rearranged) chain"),
                                      yaxis=dict(title="Displacement (nm)")))
    return fig
