# app.py — Rotational stiffness
# Auto = steepest initial prefix
# Manual = % of Mrd  OR first K points (optionally origin→point j)

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
        r"""
**Goal:** Estimate the **initial rotational stiffness** \(S_{j,ini}\) — the slope near the origin.

**Auto (recommended):**  
Looks only at the **first points by rotation**. For each prefix (first 2 points, 3 points, …) it fits a line **through the origin**, checks linearity (R²), and picks the **steepest straight** prefix before the curve bends.

**Manual options:**
- **% of Mrd:** use points with \(M \le q\% \cdot M_{rd}\).  
  This starts *lower* on the curve (early small moments) and is great when you want the first steep bit.
- **First K points:** use exactly the first **K** positive‑rotation points (as low as **K=2**).  
  (Optional) **Origin → point j**: draw the line through the origin and a chosen early point.
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

# --- Manual windows ---
def window_by_percent_of_mrd(rot, mom, mrd, q_percent, theta_min=1e-9):
    """
    Use only points with M <= q% of Mrd (not Mmax), and rotation > theta_min.
    Returns (k, r2, mask) or (None, None, mask) if too few points.
    """
    lim = (q_percent / 100.0) * float(mrd)
    mask = (mom <= lim) & (rot > theta_min)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2:
        return None, None, mask
    k = ols_slope_through_origin(xx, yy)
    if k is None or not np.isfinite(k):
        return None, None, mask
    r2 = r2_linear(yy, k * xx)
    return k, r2, mask

def manual_first_k(rot, mom, K, theta_min=1e-9):
    """Use exactly the first K positive-rotation points."""
    pos = rot > theta_min
    xx = rot[pos][:K]; yy = mom[pos][:K]
    if xx.size < 2:
        return None, None, None, "manual K: not enough points"
    k = ols_slope_through_origin(xx, yy)
    if k is None or not np.isfinite(k):
        return None, None, None, "manual K: invalid slope"
    r2 = r2_linear(yy, k * xx)
    mask = np.zeros_like(rot, dtype=bool)
    mask[np.where(pos)[0][:xx.size]] = True
    return k, r2, mask, f"manual K, first {xx.size} points"

def manual_origin_to_point(rot, mom, j_index, theta_min=1e-9):
    """
    Classical 'origin→point j' secant: slope = M_j / R_j using the j-th positive-rotation point.
    j_index is 1-based for the first positive-rotation point.
    """
    pos_idx = np.where(rot > theta_min)[0]
    if pos_idx.size < 1:
        return None, None, None, "origin→point: no positive-rotation points"
    j0 = int(j_index) - 1
    if j0 < 0 or j0 >= pos_idx.size:
        return None, None, None, "origin→point: index out of range"
    j = pos_idx[j0]
    Rx, My = rot[j], mom[j]
    if Rx == 0:
        return None, None, None, "origin→point: zero rotation at j"
    k = float(My / Rx)
    # Build a tiny mask highlighting the chosen point
    mask = np.zeros_like(rot, dtype=bool)
    mask[j] = True
    # R² doesn't really apply to 1-point secant; report n/a as 1.0
    return k, 1.0, mask, f"origin→point j={j_index}"

# --- AUTO: steepest initial prefix ---
def choose_initial_prefix(rot, mom,
                          k_cap=30,       # analyze up to the first 30 points (or all if fewer)
                          r2_min=0.995,   # required linearity
                          drop_tol=0.05,  # stop if slope drops >5% from running max
                          theta_min=1e-9):
    pos = rot > theta_min
    X = rot[pos]; Y = mom[pos]
    if X.size < 2:
        return None

    Kcap = int(min(k_cap, X.size))
    min_pts = int(max(4, min(10, round(0.10 * X.size))))  # ~10% of available, clamp [4,10]
    pos_idx = np.where(pos)[0]

    best = None; best_k = None; best_r2 = None; best_mask = None
    running_max = -np.inf; running_k = None; running_r2 = None; running_mask = None

    for k in range(2, Kcap + 1):
        xx = X[:k]; yy = Y[:k]
        k_slope = ols_slope_through_origin(xx, yy)
        if k_slope is None or not np.isfinite(k_slope):
            continue
        r2 = r2_linear(yy, k_slope * xx)

        if r2 >= r2_min and k >= min_pts and k_slope > running_max:
            best = k_slope; best_k = k; best_r2 = r2
            mask = np.zeros_like(rot, dtype=bool); mask[pos_idx[:k]] = True
            best_mask = mask

        if k_slope > running_max:
            running_max = k_slope; running_k = k; running_r2 = r2
            mask = np.zeros_like(rot, dtype=bool); mask[pos_idx[:k]] = True
            running_mask = mask
        else:
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

# ---------- Sidebar ----------
with st.sidebar:
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=340.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness")

    mode = st.radio(
        "Initial stiffness window",
        ["Auto (recommended)", "Manual (% of Mrd)", "Manual (first K points)"],
        index=0
    )

    if mode == "Manual (% of Mrd)":
        q_mrd = st.slider("q% of Mrd (window cap)", 0.5, 50.0, 5.0, 0.5)

    if mode == "Manual (first K points)":
        K_manual = st.slider("K (first points to use)", 2, 30, 2, 1)
        use_origin_point = st.checkbox("Use origin → point j instead", value=False)
        if use_origin_point:
            j_manual = st.slider("Choose point j (1 = first positive-rotation point)", 1, 30, 2, 1)

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

theta_min = 1e-9

# ---------- Compute Sj,ini ----------
if mode == "Auto (recommended)":
    choice = choose_initial_prefix(x, y, k_cap=30, r2_min=0.995, drop_tol=0.05, theta_min=theta_min)
    if choice is None:
        st.error("Could not lock onto an initial straight segment. Try a manual mode.")
        st.stop()
    Sj_ini = choice["slope"]; used_r2 = choice["r2"]; used_mask = choice["mask"]; note = choice["reason"]

elif mode == "Manual (% of Mrd)":
    k, r2, mask_q = window_by_percent_of_mrd(x, y, mrd, q_percent=q_mrd, theta_min=theta_min)
    if k is None:
        st.error("Too few points under the chosen q% of Mrd. Increase q% slightly.")
        st.stop()
    Sj_ini = k; used_r2 = r2; used_mask = mask_q; note = f"manual, M ≤ {q_mrd:g}% of Mrd"

else:  # Manual (first K points)
    if use_origin_point:
        k, r2, mask_k, note = manual_origin_to_point(x, y, j_index=j_manual, theta_min=theta_min)
    else:
        k, r2, mask_k, note = manual_first_k(x, y, K=K_manual, theta_min=theta_min)
    if k is None:
        st.error("Manual K: not enough/invalid early points. Increase K or disable origin→point.")
        st.stop()
    Sj_ini = k; used_r2 = r2; used_mask = mask_k

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

ax.plot(x_line, y_line_ini, "--", color="blue",
        label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad ({note})")
ax.plot(x_line, y_line_sj,  "--", color="green", label=f"Sj ≈ {Sj:.1f} kNm/rad")
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
c3.metric("R² (window)", f"{used_r2:.5f}" if used_r2 is not None else "n/a")
c4.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}" if x_mrd is not None else "no intersection")
