# app.py — Rotational stiffness (Auto = steepest initial prefix; Manual p% with robust fallback)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rotational Stiffness", layout="wide")
st.title("Rotational stiffness")
st.caption("Upload an Excel file with two columns: Rotation (x) and Moment (y).")

# --- Short explanation (simple) ---
with st.expander("What is happening? (tap to expand)", expanded=False):
    st.markdown(
        """
**Goal:** Estimate the **initial rotational stiffness** \\(S_{j,ini}\\) — the slope near the origin.

**Auto (recommended):**  
We only look at the **first points by rotation** (closest to zero). For each prefix (first 2 points, first 3, …) we fit a straight line **through the origin** and compute its slope and linearity (R²).  
We pick the prefix where the slope is **maximally steep** and still **linear**, stopping before the curve bends. This locks onto the **straight, steep start** of the curve.

**Manual p% (optional):**  
Use points with \(M \le p\\% \cdot M_{max}\). If there are too few points, the app automatically **expands p** until a valid window is found; if needed, it falls back to the **first K** smallest-rotation points.
        """
    )

# ---------- Small helpers ----------
def ols_slope_through_origin(x, y):
    """Slope k minimizing sum (y - kx)^2 with intercept fixed at 0."""
    xx = np.asarray(x, float); yy = np.asarray(y, float)
    denom = float(np.dot(xx, xx))
    if denom == 0:
        return None
    return float(np.dot(xx, yy) / denom)

def r2_linear(y_true, y_fit):
    """Standard R² (relative to mean of y) to check linearity."""
    y = np.asarray(y_true, float); yf = np.asarray(y_fit, float)
    ss_res = float(np.sum((y - yf) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def find_rotation_at_mrd(rot, mom, mrd):
    """Piecewise-linear crossing (last crossing if multiple)."""
    s = mom - mrd
    hits = np.where(np.isclose(s, 0.0, atol=1e-12))[0]
    if hits.size:
        return float(rot[hits[-1]])
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[-1]); x0, x1 = rot[i], rot[i+1]; y0, y1 = mom[i], mom[i+1]
    return float(x0 + (mrd - y0) * (x1 - x0) / (y1 - y0))

def window_by_percent(rot, mom, p, theta_min=1e-9):
    """(Manual mode) Points with M <= p% of Mmax & rotation > theta_min."""
    Mmax = float(np.max(mom))
    lim = (p / 100.0) * Mmax
    mask = (mom <= lim) & (rot > theta_min)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2:
        return None, None, mask
    k = ols_slope_through_origin(xx, yy)
    if k is None or not np.isfinite(k):
        return None, None, mask
    r2 = r2_linear(yy, k * xx)
    return k, r2, mask

def manual_window_or_expand(rot, mom, p_start, theta_min=1e-9):
    """
    Manual p% that always tries to succeed:
      1) Try requested p%. If <2 points, expand p by 1% steps up to 100%.
      2) If still too few points, fall back to first K=3..10 smallest-rotation points.
    Returns (k, r2, mask, note) or (None, None, None, note).
    """
    # 1) Expand p% upwards until a valid window exists
    for p_try in list(np.arange(float(p_start), 100.0 + 1e-9, 1.0)):
        k, r2, mask = window_by_percent(rot, mom, p_try, theta_min=theta_min)
        if k is not None:
            note = f"manual, p≤{p_try:g}%"
            if p_try > p_start:
                note += f" (expanded from {p_start:g}%)"
            return k, r2, mask, note

    # 2) Fallback: first K points by smallest rotation
    pos = rot > theta_min
    for K in range(3, 11):  # 3..10
        xx = rot[pos][:K]; yy = mom[pos][:K]
        if xx.size >= 2:
            k = ols_slope_through_origin(xx, yy)
            if k is not None and np.isfinite(k):
                r2 = r2_linear(yy, k * xx)
                mask = np.zeros_like(rot, dtype=bool)
                mask[np.where(pos)[0][:xx.size]] = True
                return k, r2, mask, f"manual fallback, first K={xx.size}"

    return None, None, None, "manual: could not form a valid window"

# ---------- AUTO: steepest initial prefix ----------
def choose_initial_prefix(rot, mom,
                          k_cap=30,       # analyze up to the first 30 points (or all if fewer)
                          r2_min=0.995,   # required linearity
                          drop_tol=0.05,  # stop if slope drops >5% from running max
                          theta_min=1e-9):
    """
    Iterate k = 2..Kcap (prefix length). For each prefix (first k positive-rotation points):
      - fit slope through origin,
      - compute R²,
      - track running max slope (steepest prefix).
    Return the prefix just BEFORE the first significant slope drop (> drop_tol),
    provided its R² >= r2_min and k >= min_pts. If no drop found, return the
    steepest prefix that meets R²; else running max as last resort.
    """
    pos = rot > theta_min
    X = rot[pos]; Y = mom[pos]
    if X.size < 2:
        return None

    Kcap = int(min(k_cap, X.size))
    min_pts = int(max(4, min(10, round(0.10 * X.size))))  # ~10% of available, clamp [4,10]

    best = None; best_k = None; best_r2 = None; best_mask = None
    running_max = -np.inf; running_k = None; running_r2 = None; running_mask = None

    pos_idx = np.where(pos)[0]

    for k in range(2, Kcap + 1):
        xx = X[:k]; yy = Y[:k]
        k_slope = ols_slope_through_origin(xx, yy)
        if k_slope is None or not np.isfinite(k_slope):
            continue
        r2 = r2_linear(yy, k_slope * xx)

        if r2 >= r2_min and k >= min_pts and k_slope > running_max:
            best = k_slope; best_k = k; best_r2 = r2
            mask = np.zeros_like(rot, dtype=bool)
            mask[pos_idx[:k]] = True
            best_mask = mask

        if k_slope > running_max:
            running_max = k_slope; running_k = k; running_r2 = r2
            mask = np.zeros_like(rot, dtype=bool)
            mask[pos_idx[:k]] = True
            running_mask = mask
        else:
            # first significant drop from the max? choose the max just before the bend
            if running_max > 0 and (running_max - k_slope) / running_max >= drop_tol:
                if running_r2 is not None and running_k is not None and running_r2 >= r2_min and running_k >= min_pts:
                    return {"k": running_k, "slope": running_max, "r2": running_r2,
                            "mask": running_mask, "reason": f"auto prefix: steepest before bend (k={running_k})"}
                if best is not None:
                    return {"k": best_k, "slope": best, "r2": best_r2,
                            "mask": best_mask, "reason": f"auto prefix: best R² & steep (k={best_k})"}

    if best is not None:
        return {"k": best_k, "slope": best, "r2": best_r2,
                "mask": best_mask, "reason": f"auto prefix: best R² & steep (k={best_k})"}
    if running_k is not None:
        return {"k": running_k, "slope": running_max, "r2": running_r2,
                "mask": running_mask, "reason": f"auto prefix: running max (k={running_k})"}
    return None

# ---------- Sidebar (minimal controls) ----------
with st.sidebar:
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=1590.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness")
    mode = st.radio("Initial stiffness window", ["Auto (recommended)", "Manual p%"], index=0)
    if mode == "Manual p%":
        p_manual = st.slider("p% of peak M", 0.5, 20.0, 2.0, 0.5)

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

# ---------- Load & tidy (assume exactly 2 columns: Rotation, Moment) ----------
df = pd.read_excel(uploaded)
x = df.iloc[:, 0].to_numpy(float)
y = df.iloc[:, 1].to_numpy(float)
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]; y = y[mask]
order = np.argsort(x)
x = x[order]; y = y[order]

# ---------- Compute Sj,ini ----------
theta_min = 1e-9

if mode == "Auto (recommended)":
    choice = choose_initial_prefix(x, y, k_cap=30, r2_min=0.995, drop_tol=0.05, theta_min=theta_min)
    if choice is None:
        st.error("Could not lock onto an initial straight segment. Try Manual p% (e.g., 1–3%).")
        st.stop()
    Sj_ini = choice["slope"]; used_r2 = choice["r2"]; used_mask = choice["mask"]; note = choice["reason"]
else:
    k, r2, mask_p, note = manual_window_or_expand(x, y, p_start=p_manual, theta_min=theta_min)
    if k is None:
        st.error("Manual mode: could not form a valid window. Try a larger p% or check your data.")
        st.stop()
    Sj_ini = k; used_r2 = r2; used_mask = mask_p

Sj = Sj_ini / 2.0

# ---------- Mrd & stiffness lines ----------
x_mrd = find_rotation_at_mrd(x, y, mrd)
x_max = float(np.max(x)) if x.size else 0.0
x_end_ini = (mrd / Sj_ini) if Sj_ini else x_max
x_end_sj  = (mrd / Sj)     if Sj     else x_max
limits = [v for v in [x_end_ini, x_end_sj, (x_mrd if x_mrd is not None else x_max), x_max] if v is not None and np.isfinite(v)]
x_limit = float(np.min(limits)) if limits else x_max

x_line = np.array([0.0, x_limit], float)
y_line_ini = Sj_ini * x_line
y_line_sj  = Sj * x_line

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, y, label="Moment–Rotation", color="black", linewidth=1.8)

if used_mask is not None and used_mask.any():
    ax.scatter(x[used_mask], y[used_mask], s=28, edgecolor="blue", facecolor="none",
               linewidth=1.2, label="Points used for Sj,ini")

legend_note = note if mode == "Manual p%" else note
ax.plot(x_line, y_line_ini, "--", color="blue",
        label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad ({legend_note})")
ax.plot(x_line, y_line_sj,  "--", color="green", label=f"Sj ≈ {Sj:.1f} kNm/rad")
ax.axhline(mrd, linestyle=":", color="red", label=f"Mrd = {mrd:.1f} kNm")
if x_mrd is not None:
    ax.plot(x_mrd, mrd, "ro", label="Mrd point")
else:
    ax.text(0.02, 0.95, "Warning: Mrd not crossed within data range",
            transform=ax.transAxes, color="red", va="top", fontsize=9)

ax.set_xlabel("Rotation [rad]")
ax.set_ylabel("Moment_
