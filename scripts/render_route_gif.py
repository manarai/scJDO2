"""Render a rotating animated GIF of a ``scjdo2 simulate`` route page.

GitHub READMEs cannot run the page's 3-D viewer, so the front page shows
this GIF and links to the interactive page. The GIF is made from the page
itself: a copy gets a small capture hook (frame and camera angle from the
URL), headless Google Chrome screenshots each frame, and Pillow joins them.
Walkers advance from the first to the last frame while the camera turns
once around the landscape.

Usage::

    python scripts/render_route_gif.py --page results/route_page.html \
        --out docs/gallery/palantir_routes.gif

Needs Google Chrome and Pillow; uses the page's CDN scripts (network).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
START = "applyTheme(); resize(); setFrame(fr); requestAnimationFrame(tick);"
HOOK = """applyTheme(); resize();
  {
    const q = new URLSearchParams(location.search);
    const az = parseFloat(q.get('az') || '0') * Math.PI / 180;
    playing = false; fr = Math.round(Math.min(1, parseFloat(q.get('f') || '0')) * (F - 1)); setFrame(fr);
    if (q.get('theme')) document.documentElement.setAttribute('data-theme', q.get('theme'));
    const zoom = parseFloat(q.get('zoom') || '1'), tilt = parseFloat(q.get('tilt') || '1');
    const t = controls.target, p = camera.position.clone().sub(t).multiplyScalar(zoom), c = Math.cos(az), s = Math.sin(az);
    p.y *= tilt;
    camera.position.set(t.x + p.x * c - p.z * s, t.y + p.y, t.z + p.x * s + p.z * c);
    controls.enableDamping = false; controls.update();
    if (q.get('real') === '1') { const r = document.getElementById('optReal'); if (r && !r.checked) r.click(); }
  }
  requestAnimationFrame(tick);"""
CSS = """<style>
  body * { visibility: hidden !important; }
  #stage, #stage * { visibility: visible !important; }
  #stage .hint { display: none !important; }
  #stage { position: fixed !important; inset: 0 !important; width: 100vw !important;
           height: 100vh !important; margin: 0 !important; border-radius: 0 !important; z-index: 9999; }
</style>
"""


def capture_page(page: Path, dst: Path) -> None:
    html = page.read_text()
    if html.count(START) != 1:
        raise SystemExit(f"{page}: viewer start line not found; is this a scjdo2 route page?")
    dst.write_text(CSS + html.replace(START, HOOK))


def shoot(cap: Path, f: float, az: float, out: Path, w: int, h: int, budget: int, view: str = "") -> Path:
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--user-data-dir={prof}", f"--window-size={w},{h}", f"--virtual-time-budget={budget}",
                        f"--screenshot={out}", f"file://{cap}?f={f:.4f}&az={az:.2f}{view}"],
                       check=True, capture_output=True, timeout=120)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--page", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--turn", type=float, default=360.0, help="degrees the camera turns over the GIF")
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--colors", type=int, default=256)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--budget-ms", type=int, default=5000, help="Chrome virtual-time budget per frame")
    p.add_argument("--theme", choices=["light", "dark"], default="dark")
    p.add_argument("--zoom", type=float, default=1.12, help="camera distance factor (< 1 = closer)")
    p.add_argument("--tilt", type=float, default=0.55, help="camera height factor (< 1 = lower view)")
    p.add_argument("--no-real", action="store_true", help="hide the measured cells")
    p.add_argument("--test-az", type=float, default=0.0)
    p.add_argument("--test", action="store_true", help="render one frame to --out (PNG) and stop")
    a = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="routegif_"))
    try:
        cap = tmp / "capture.html"
        capture_page(a.page.resolve(), cap)
        view = f"&theme={a.theme}&zoom={a.zoom}&tilt={a.tilt}&real={0 if a.no_real else 1}"
        if a.test:
            shoot(cap, 0.6, a.test_az, a.out.resolve(), a.width, a.height, a.budget_ms, view)
            print(f"wrote {a.out}")
            return
        n = a.frames
        jobs = [(cap, i / (n - 1), a.turn * i / n, tmp / f"f{i:03d}.png") for i in range(n)]
        with ThreadPoolExecutor(a.jobs) as ex:
            pngs = list(ex.map(lambda j: shoot(*j, a.width, a.height, a.budget_ms, view), jobs))
        frames = [Image.open(x).convert("RGB") for x in pngs]
        # one palette for all frames, from a strip of frames spread over the GIF
        pick = [frames[i] for i in range(0, n, max(1, n // 6))]
        strip = Image.new("RGB", (a.width, a.height * len(pick)))
        for k, im in enumerate(pick):
            strip.paste(im, (0, k * a.height))
        pal = strip.quantize(colors=a.colors, method=Image.Quantize.MAXCOVERAGE)
        q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        q[0].save(a.out, save_all=True, append_images=q[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
        print(f"wrote {a.out} ({n} frames, {a.out.stat().st_size / 1e6:.1f} MB)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
