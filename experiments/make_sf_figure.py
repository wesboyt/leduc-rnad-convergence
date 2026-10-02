"""make_sf_figure.py -- animated SVG: schedule-free is stable in a gradient field and unstable in a rotational one.

Both panels run the REAL optimizers (torch SGD and schedulefree.SGDScheduleFree, beta 0.9, no warmup) on the 2-D
linearised regularised game F(w) = (lambda I + omega J) w, lambda = 1, from the same start, with the same step 0.05:
  left   omega = 0      a gradient field (minimisation): both optimizers converge
  right  omega = 3      a rotational field (a regularised game): SGD spirals in, schedule-free spirals out
Animation time is proportional to optimizer steps. Claim C22 is the 16-dimensional, pre-registered version.

    python experiments/make_sf_figure.py          # writes docs/sf_fields.svg
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrc.policy import make_optimizer  # noqa: E402

LAM, GAMMA, BETA, STEPS, START = 1.0, 0.05, 0.9, 150, (1.6, 0.9)
LIM = 3.2                     # plot range [-LIM, LIM]^2
PANEL, PAD, TOP = 440, 40, 96
W, H = 2 * PANEL + 3 * PAD, TOP + PANEL + 120
DUR = 9.0                     # seconds per loop: 6 s of motion, 3 s hold
COL = {"sgd": "#2563eb", "sfsgd": "#ea580c"}
NAME = {"sgd": "SGD", "sfsgd": "schedule-free SGD (β = 0.9)"}


def trajectory(opt, omega):
    p = torch.nn.Parameter(torch.tensor(START, dtype=torch.float64))
    o = make_optimizer([p], opt, GAMMA, betas=(BETA, 0.999), warmup=0)
    pts = []
    for _ in range(STEPS + 1):
        x, y = (float(v) for v in p.detach())
        pts.append((x, y))
        if math.hypot(x, y) > 4 * LIM:          # far outside the frame: stop recording
            break
        o.zero_grad()
        p.grad = torch.tensor([LAM * x + omega * y, -omega * x + LAM * y], dtype=torch.float64)
        o.step()
    return pts


def to_px(x, y, x0):
    return x0 + (x + LIM) / (2 * LIM) * PANEL, TOP + (LIM - y) / (2 * LIM) * PANEL


def path_and_keys(pts, x0):
    px = [to_px(x, y, x0) for x, y in pts]
    seg = [math.dist(px[i], px[i + 1]) for i in range(len(px) - 1)]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1]) if cum[-1] > 0 else 1.0
    d = "M" + " L".join("%.1f,%.1f" % q for q in px)
    n = len(px) - 1
    frac_steps = n / STEPS                          # a run that leaves the frame early still moves at step pace
    kt_move = [0.6667 * frac_steps * i / n for i in range(n + 1)]
    kt = kt_move + [1.0]
    kp = [c / L for c in cum] + [1.0]
    return d, L, kt, kp


def quiver(omega, x0):
    out = []
    g = np.linspace(-LIM * 0.85, LIM * 0.85, 9)
    for x in g:
        for y in g:
            fx, fy = -(LAM * x + omega * y), -(-omega * x + LAM * y)     # the flow the optimizer follows (-F)
            n = math.hypot(fx, fy)
            if n < 1e-9:
                continue
            ux, uy = fx / n, fy / n
            (sx, sy) = to_px(x, y, x0)
            ex, ey = sx + 17 * ux, sy - 17 * uy
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="q" marker-end="url(#ah)"/>' % (
                sx, sy, ex, ey))
    return "\n".join(out)


def panel(omega, x0, title, subtitle, cid):
    parts = ['<rect x="%d" y="%d" width="%d" height="%d" class="frame"/>' % (x0, TOP, PANEL, PANEL),
             '<text x="%d" y="%d" class="t">%s</text>' % (x0, TOP - 34, title),
             '<text x="%d" y="%d" class="s">%s</text>' % (x0, TOP - 14, subtitle),
             '<g clip-path="url(#%s)">' % cid, quiver(omega, x0)]
    cx, cy = to_px(0, 0, x0)
    parts.append('<circle cx="%.1f" cy="%.1f" r="6" class="eq"/>' % (cx, cy))
    ends = {}
    for opt in ("sgd", "sfsgd"):
        pts = trajectory(opt, omega)
        d, L, kt, kp = path_and_keys(pts, x0)
        ends[opt] = pts
        kts = ";".join("%.4f" % k for k in kt)
        offs = ";".join("%.1f" % (L * (1 - k)) for k in kp)
        parts.append(
            '<path d="%s" fill="none" stroke="%s" stroke-width="2.4" stroke-linejoin="round" '
            'stroke-dasharray="%.1f" stroke-dashoffset="%.1f">'
            '<animate attributeName="stroke-dashoffset" dur="%ss" repeatCount="indefinite" calcMode="linear" '
            'keyTimes="%s" values="%s"/></path>' % (d, COL[opt], L, L, DUR, kts, offs))
        parts.append(
            '<circle r="6" fill="%s" stroke="#fff" stroke-width="1.5">'
            '<animateMotion dur="%ss" repeatCount="indefinite" calcMode="linear" keyTimes="%s" keyPoints="%s" '
            'path="%s"/></circle>' % (COL[opt], DUR, kts, ";".join("%.4f" % k for k in kp), d))
    parts.append("</g>")
    sx, sy = to_px(*START, x0)
    parts.append('<circle cx="%.1f" cy="%.1f" r="4" class="start"/>' % (sx, sy))
    parts.append('<text x="%.1f" y="%.1f" class="lab">start</text>' % (sx + 8, sy - 8))
    parts.append('<text x="%.1f" y="%.1f" class="lab">equilibrium</text>' % (cx + 9, cy + 18))
    return "\n".join(parts), ends


def outcome(pts):
    r = math.hypot(*pts[-1])
    exit_step = next((i for i, (x, y) in enumerate(pts) if max(abs(x), abs(y)) > LIM), None)
    return r, exit_step


def main():
    xl, xr = PAD, 2 * PAD + PANEL
    left, el = panel(0.0, xl, "Gradient field (ω = 0)", "minimisation: the regime schedule-free was built for", "cl")
    right, er = panel(3.0, xr, "Rotational field (ω = 3λ)", "a regularised two-player game near its equilibrium", "cr")
    rl_sgd, _ = outcome(el["sgd"])
    rl_sf, _ = outcome(el["sfsgd"])
    rr_sgd, _ = outcome(er["sgd"])
    rr_sf, n_sf = outcome(er["sfsgd"])
    yb = TOP + PANEL + 34
    cap_l = ("Both converge.", "after %d steps: |w| = %.3f (SGD), %.3f (schedule-free)" % (STEPS, rl_sgd, rl_sf))
    cap_r = ("SGD spirals in; schedule-free spirals out.",
             "SGD |w| = %.3f after %d steps; schedule-free leaves the frame at step %d (|w| = %.1f at %d)" % (
                 rr_sgd, STEPS, n_sf, rr_sf, STEPS))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
 aria-label="Animated comparison: SGD and schedule-free SGD both converge in a gradient field; in a rotational field SGD converges and schedule-free SGD diverges.">
<title>Schedule-free is stable in gradient fields and unstable in rotational ones</title>
<style>
  .bg {{ fill: #ffffff; }}
  .frame {{ fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; }}
  .q {{ stroke: #94a3b8; stroke-width: 1.2; }}
  .t {{ font: 600 19px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0f172a; }}
  .s {{ font: 400 13.5px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #475569; }}
  .lab {{ font: 400 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #334155; }}
  .cap {{ font: 400 11.5px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #334155; }}
  .leg {{ font: 500 13.5px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0f172a; }}
  .foot {{ font: 400 11.5px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #64748b; }}
  .eq {{ fill: #0f172a; }}
  .start {{ fill: #ffffff; stroke: #0f172a; stroke-width: 1.5; }}
</style>
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker>
  <clipPath id="cl"><rect x="{xl}" y="{TOP}" width="{PANEL}" height="{PANEL}"/></clipPath>
  <clipPath id="cr"><rect x="{xr}" y="{TOP}" width="{PANEL}" height="{PANEL}"/></clipPath>
</defs>
<rect class="bg" width="{W}" height="{H}"/>
<g>
  <line x1="{PAD}" y1="22" x2="{PAD + 26}" y2="22" stroke="{COL["sgd"]}" stroke-width="3"/>
  <text x="{PAD + 34}" y="27" class="leg">{NAME["sgd"]}</text>
  <line x1="{PAD + 110}" y1="22" x2="{PAD + 136}" y2="22" stroke="{COL["sfsgd"]}" stroke-width="3"/>
  <text x="{PAD + 144}" y="27" class="leg">{NAME["sfsgd"]}</text>
  <text x="{PAD + 380}" y="27" class="lab">grey arrows: the flow −F(w) each optimizer follows</text>
</g>
{left}
{right}
<text x="{xl}" y="{yb}" class="leg">{cap_l[0]}</text>
<text x="{xl}" y="{yb + 19}" class="cap">{cap_l[1]}</text>
<text x="{xr}" y="{yb}" class="leg">{cap_r[0]}</text>
<text x="{xr}" y="{yb + 19}" class="cap">{cap_r[1]}</text>
<text x="{PAD}" y="{H - 42}" class="foot">F(w) = (λI + ωJ)w with λ = 1 (the regulariser's pull) and ω the game's rotation. Same start, same step γ = {GAMMA}, {STEPS} steps; real torch SGD and</text>
<text x="{PAD}" y="{H - 24}" class="foot">schedulefree.SGDScheduleFree. Schedule-free evaluates the gradient at y = (1−β)z + βx, pulled toward its running average x, which lags behind a rotating iterate.</text>
</svg>
'''
    out = os.path.join(ROOT, "docs", "sf_fields.svg")
    open(out, "w", encoding="utf-8", newline="\n").write(svg)
    print("wrote", out)
    print(*cap_l)
    print(*cap_r)


if __name__ == "__main__":
    main()
