"""
Two 3D viewports whose cameras move together (rotate / zoom / pan one, the other follows).

Plotly has no built-in camera linking across figures, so both figures are drawn by plotly.js in
one small HTML component that copies `scene.camera` from the view being dragged to the other on
every relayout event (throttled to animation frames). plotly.js itself is served from the app's
own static folder (`static/`, enabled in .streamlit/config.toml), so it works offline; if static
serving is unavailable the component falls back to the public CDN.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path

import plotly
import plotly.graph_objects as go
import plotly.offline as po
import streamlit as st

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
JS_VERSION = po.get_plotlyjs_version()
JS_NAME = f"plotly-{JS_VERSION}.min.js"
CDN = f"https://cdn.plot.ly/plotly-{JS_VERSION}.min.js"


def ensure_plotly_js() -> str:
    """Write the plotly.js bundled with the Python package into static/ (once) and return its URL path."""
    path = STATIC_DIR / JS_NAME
    if not path.exists():
        try:
            STATIC_DIR.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(po.get_plotlyjs(), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            return CDN
    return f"app/static/{JS_NAME}"


def _fig_json(fig: go.Figure) -> str:
    # JSON is valid JavaScript; "</" is escaped so no string inside can close the <script> element.
    return fig.to_json().replace("</", "<\\/")


def render_pair(left: go.Figure, right: go.Figure, left_title: str, right_title: str, height: int = 560,
                key_hint: str = "") -> None:
    src = ensure_plotly_js()
    config = {"displaylogo": False, "responsive": True, "scrollZoom": True,
              "modeBarButtonsToRemove": ["zoom3d", "pan3d", "orbitRotation", "tableRotation", "handleDrag3d",
                                         "resetCameraLastSave3d", "hoverClosest3d"],
              "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "chronocell_compare"}}
    doc = f"""
<!doctype html><html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: transparent; font-family: 'Inter Tight', system-ui, sans-serif; }}
  .wrap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .t {{ font: 500 12px/1.4 'Inter Tight', system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase;
        color: #62645F; padding: 2px 0 6px; border-bottom: 1px solid #D8D8D3; }}
  .t b {{ color: #1C1E1B; font-weight: 600; }}
  .v {{ height: {height}px; border-bottom: 1px solid #D8D8D3; }}
  .msg {{ font: 13px system-ui; color: #7A3E12; padding: 12px; }}
  .sync {{ font: 400 11.5px system-ui; color: #62645F; padding-top: 6px; }}
</style></head><body>
<div class="wrap">
  <div><div class="t">Left · <b>{_html.escape(left_title)}</b></div><div id="a" class="v"></div></div>
  <div><div class="t">Right · <b>{_html.escape(right_title)}</b></div><div id="b" class="v"></div></div>
</div>
<div class="sync">Cameras are linked: rotate, zoom or pan either view and the other follows. {_html.escape(key_hint)}</div>
<script src="{src}"></script>
<script>
(function () {{
  function start() {{
    const A = {_fig_json(left)}, B = {_fig_json(right)}, cfg = {json.dumps(config)};
    Promise.all([Plotly.newPlot('a', A.data, A.layout, cfg), Plotly.newPlot('b', B.data, B.layout, cfg)])
      .then(function (g) {{
        let busy = false, pending = null;
        function link(src, dst) {{
          const copy = function (ev) {{
            const cam = ev && ev['scene.camera'];
            if (!cam || busy) return;
            pending = cam;
            window.requestAnimationFrame(function () {{
              if (!pending) return;
              busy = true;
              Plotly.relayout(dst, {{'scene.camera': pending}}).then(function () {{ busy = false; }});
              pending = null;
            }});
          }};
          src.on('plotly_relayouting', copy);
          src.on('plotly_relayout', copy);
        }}
        link(g[0], g[1]); link(g[1], g[0]);
        window.__ccSync = g;   // for automated checks
      }});
  }}
  if (window.Plotly) {{ start(); return; }}
  const s = document.createElement('script');
  s.src = {json.dumps(CDN)};
  s.onload = start;
  s.onerror = function () {{ document.body.innerHTML = '<div class="msg">Could not load plotly.js for the linked view.</div>'; }};
  document.head.appendChild(s);
}})();
</script></body></html>"""
    # The document is generated here from our own figures (file-derived titles are escaped).
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height + 70)
    else:  # Streamlit versions before st.iframe
        import streamlit.components.v1 as components
        components.html(doc, height=height + 70)


def plotly_version() -> str:
    return f"{plotly.__version__} (plotly.js {JS_VERSION})"
