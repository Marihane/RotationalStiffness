# -*- coding: utf-8 -*-
"""
Rotational stiffness plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================== Inputs ========================================
FILE = Path(__file__).with_name("Example.xlsx")   # Excel file in same folder as script
MRD = 1590                                        # Design moment kNm
TITLE = 'Rotational Stiffness - Example'          # Graph title
I_TANGENT = 3                                     # Index for Sj,ini steepness
# ===========================================================================


def rot_at_mrd(rot, mom, target):
    """Return rotation where the polyline mom(rot) crosses target.
    Works when the curve is not monotonic. Picks the last crossing."""
    rot = np.asarray(rot, dtype=float)
    mom = np.asarray(mom, dtype=float)

    hits = np.where(np.isclose(mom, target, atol=1e-9))[0]
    if hits.size:
        return rot[hits[-1]]

    s = mom - target
    idx = np.where((s[:-1] == 0) | (s[:-1] * s[1:] < 0) | (s[1:] == 0))[0]
    if idx.size == 0:
        return None

    i = idx[-1]
    x0, x1 = rot[i], rot[i + 1]
    y0, y1 = mom[i], mom[i + 1]
    if y1 == y0:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def safe_min(*vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return min(vals) if vals else None


# Load data
if not FILE.exists():
    raise FileNotFoundError(f"Excel file not found at: {FILE}")

df = pd.read_excel(FILE)
rotation = df['Rotation'].to_numpy(dtype=float)
moment = df['Moment'].to_numpy(dtype=float)

if rotation.size < 2 or moment.size < 2:
    raise ValueError('Need at least two data points')
if not (1 <= I_TANGENT < rotation.size):
    raise ValueError('I_TANGENT must be between 1 and len(data)-1')

# Stiffness
x_tan = rotation[I_TANGENT]
y_tan = moment[I_TANGENT]
if x_tan == 0:
    raise ValueError('Rotation at I_TANGENT is zero')

Sj_ini = y_tan / x_tan
Sj = Sj_ini / 2.0

# Mrd intersection and limits for stiffness lines
x_mrd = rot_at_mrd(rotation, moment, MRD)
x_max = np.nanmax(rotation)
x_lim_ini = MRD / Sj_ini if Sj_ini else None
x_lim_sj = MRD / Sj if Sj else None
x_limit = safe_min(x_lim_ini, x_lim_sj, x_mrd if x_mrd is not None else x_max, x_max)

x_line = np.array([0.0, x_limit])
y_line_ini = Sj_ini * x_line
y_line_sj = Sj * x_line

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(rotation, moment, label='Moment–Rotation', color='black', linewidth=1.8)
ax.plot(x_line, y_line_ini, '--', label=f'Sj,ini ≈ {Sj_ini:.1f} kNm/rad', color='blue')
ax.plot(x_line, y_line_sj, '--', label=f'Sj ≈ {Sj:.1f} kNm/rad', color='green')
ax.axhline(MRD, linestyle=':', color='red', label=f'Mrd = {MRD:.1f} kNm')

if x_mrd is not None and np.isfinite(x_mrd):
    ax.plot(x_mrd, MRD, 'ro', label='Mrd point')
else:
    print('Warning: Mrd is outside the curve range so the red point is not plotted')

ax.set_xlabel('Rotation [rad]')
ax.set_ylabel('Moment [kNm]')
ax.set_title(TITLE)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend()
fig.tight_layout()
plt.show()

# Summary
print(f"Sj,ini = {Sj_ini:.1f} kNm/rad")
print(f"Sj = {Sj:.1f} kNm/rad")
if x_mrd is not None:
    print(f"Mrd = {MRD:.1f} kNm occurs at rotation ≈ {x_mrd:.6f} rad")
else:
    print(f"Mrd = {MRD:.1f} kNm does not intersect the curve within the data range")
