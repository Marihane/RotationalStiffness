# app.py — Rotational stiffness (simple, adaptive auto-window)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rotational Stiffness", layout="wide")
st.title("Rotational stiffness")
st.caption("Upload an Excel file with two columns: Rotation (x), Moment (y)")

# ---------- Small helpers ----------
def ols_slope_through_origin(x, y):
    """Slope k minimizing sum (y - kx)^2 with intercept fixed at 0."""
    xx = np.asarray(x, float)
    yy = np.asarray(y, float)
    denom = np.dot(xx, xx)
    if denom == 0:
        return None
    return float(np.dot(xx, yy) / denom)

def r2_linear(y_true, y_fit):
    """Standard R² (w.r.t. mean of y) to check linearity of the window."""
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    ss_res = np.sum((y_true - y_fit)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0

def find_rotation_at_mrd(rot, mom, mrd):
    """
    Return the rotation where the piecewise-linear curve crosses Mrd.
    Picks the last crossing if there are several. Returns None if no crossing.
    """
    s = mom - mrd
    # exact hit
    hits = np.where(np.isclose(s, 0.0, atol=1e-12))[0]
    if hits.size:
        return float(rot[hits[-1]])
    # sign change between successive points => crossing on that segment
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[-1])
    x0, x1 = rot[i], rot[i + 1]
    y0, y1 = mom[i], mom[i + 1]
    # linear interpolation on the segment
    return float(x0 + (mrd - y0) * (x1 - x0) / (y1 - y0))

def window_slope_by_percent(rot, mom, p):
    """
    Use only points with M <= p% of peak M, skip exact zero rotation,
    fit slope through origin, and return (k, r2, mask).
    """
    Mmax = float(np.max(mom))
    lim = (p / 100.0) * Mmax
    mask = (mom <= lim) & (rot > 0.0)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2:
        return None, None, mask
    k = ols_slope_through_origin(xx, yy)
    yfit = k * xx
    r2 = r2_linear(yy, yfit)
    return k, r2, mask

def choose_auto_window(rot, mom,
                       p_min=2, p_max=25,
                       r2_strict=0.995, r2_loose=0.99):
    """
    Adaptive auto window:
      - Try p in [p_min, p_max] (% of peak M).
      - Adaptive min points ≈ 15% of N, clamped to [6,12].
      - Prefer the LARGEST p with R² >= r2_strict and enough points.
      - If none, accept R² >= r2_loose.
      - Fallback = highest R² with at least 3 points.
    Returns {p, k, r2, mask} or None.
    """
    N = len(rot)
    min_pts = int(max(6, min(12, round(0.15 * N))))

    strict_ok = None
    loose_ok = None
    fallback = None

    for p in range(int(p_min), int(p_max) + 1):
        k, r2, mask = window_slope_by_percent(rot, mom, p)
        if k is None:
            continue
        n = int(mask.sum())

        # Best overall R² (fallback), require ≥3 points
        if n >= 3 and (fallback is None or r2 > fallback["r2"]):
            fallback = {"p": p, "k": k, "r2": r2, "mask": mask}

        # Keep largest p that meets strict criteria
        if n >= min_pts and r2 >= r2_strict:
            strict_ok = {"p": p, "k": k, "r2": r2, "mask": mask}

        # Keep largest p that meets loose criteria
        if n >= min_pts and r2 >= r2_loose:
            loose_ok = {"p": p, "k": k, "r2": r2, "mask": mask}

    if strict_ok is not None:
        strict_ok["reason"] = "auto (strict)"
        return strict_ok
    if loose_ok is not None:
        loose_ok["reason"] = "auto (loose)"
        return loose_ok
    if fallback is not None:
        fallback["reason"] = "auto (fallback)"
        return fallback
    return None

# ---------- Sidebar (minimal controls) ----------
with st.sidebar:
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=1590.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness")
    mode = st.radio("Initial stiffness window", ["Auto (recommended)", "Manual p%"], index=0)
    if mode == "Manual p%":
        p_manual = st.slider("p% of peak M", 5, 25, 10, 1)

# ---------- File upload ----------
uploaded = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    help="Two columns: Rotation (x), Moment (y). First row can be header."
)

if not uploaded:
    st.info("Upload an Excel file to begin")
    st.stop()

# ---------- Load data (assume exactly 2 columns: Rotation, Moment) ----------
df = pd.read_excel(uploaded)
x = df.iloc[:, 0].to_numpy(float)  # Rotation
y = df.iloc[:, 1].to_numpy(float)  # Moment

# Tidy: drop NaNs, sort by rotation
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]; y = y[mask]
order = np.argsort(x)
x = x[order]; y = y[order]

# ---------- Compute Sj,ini ----------
if mode == "Auto (recommended)":
    choice = choose_auto_window(x, y)  # adaptive defaults
    if choice is None:
        st.error("Could not determine a stable initial window. Check your data.")
        st.stop()
    Sj_ini = choice["k"]
    used_r2 = choice["r2"]
    used_mask = choice["mask"]
    note = f"{choice['reason']}, p≤{choice['p']}%"
else:
    k, r2, mask_p = window_slope_by_percent(x, y, p_manual)
    if k is None:
        st.error("Not enough points in the selected window. Increase p%.")
        st.stop()
    Sj_ini = k
    used_r2 = r2
    used_mask = mask_p
    note = f"manual, p≤{p_manual}%"

Sj = Sj_ini / 2.0

# ---------- Mrd & stiffness lines ----------
x_mrd = find_rotation_at_mrd(x, y, mrd)
x_max = float(np.max(x)) if x.size else 0.0
x_end_ini = (mrd / Sj_ini) if Sj_ini != 0 else x_max
x_end_sj  = (mrd / Sj)     if Sj     != 0 else x_max

# Clip stiffness lines so they don't overshoot the curve range or Mrd
limits = [v for v in [x_end_ini, x_end_sj, (x_mrd if x_mrd is not None else x_max), x_max] if np.isfinite(v)]
x_limit = float(np.min(limits)) if limits else x_max

x_line = np.array([0.0, x_limit], float)
y_line_ini = Sj_ini * x_line
y_line_sj  = Sj * x_line

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, y, label="Moment–Rotation", color="black", linewidth=1.8)

# Highlight points used for Sj,ini
if used_mask is not None and used_mask.any():
    ax.scatter(x[used_mask], y[used_mask], s=28, edgecolor="blue", facecolor="none",
               linewidth=1.2, label="Points used for Sj,ini")

ax.plot(x_line, y_line_ini, "--", color="blue",
        label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad ({note})")
ax.plot(x_line, y_line_sj,  "--", color="green",
        label=f"Sj ≈ {Sj:.1f} kNm/rad")
ax.axhline(mrd, linestyle=":", color="red", label=f"Mrd = {mrd:.1f} kNm")
if x_mrd is not None:
    ax.plot(x_mrd, mrd, "ro", label="Mrd point")
else:
    ax.text(0.02, 0.95, "Warning: Mrd not crossed within data range",
            transform=ax.transAxes, color="red", va="top", fontsize=9)

ax.set_xlabel("Rotation [rad]")
ax.set_ylabel("Moment [kNm]")
ax.set_title(title)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend()
fig.tight_layout()
st.pyplot(fig, clear_figure=False)

# ---------- Results ----------
st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sj,ini [kNm/rad]", f"{Sj_ini:.1f}")
c2.metric("Sj [kNm/rad]", f"{Sj:.1f}")
c3.metric("R² (window)", f"{used_r2:.5f}")
c4.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}" if x_mrd is not None else "no intersection")
