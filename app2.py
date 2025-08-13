# app.py — Rotational stiffness (initial-slope biased auto-window)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rotational Stiffness", layout="wide")
st.title("Rotational stiffness")

st.caption("Upload an Excel file with two columns: Rotation (x) and Moment (y).")

# --- Short explanation ---
with st.expander("What is happening? (tap to expand)", expanded=False):
    st.markdown(
        """
**Goal:** Estimate the **initial rotational stiffness** \\(S_{j,ini}\\): the slope near the origin.

**Auto mode (recommended):**
- We look only at the **very early part** of the curve where moment is small.
- We try windows up to **p% of the peak moment** with **p from 0.5% to 12%**.
- For each window we fit a straight line **through the origin** and check linearity (R²) and slope stability.
- We pick the **smallest p** that is still linear and stable — to stay truly *initial*.

**What is p%?**  
If the peak moment is 1000 kNm and **p = 2%**, we only use points with **M ≤ 20 kNm** to compute the slope.
        """
    )

# ---------- Helpers ----------
def ols_slope_through_origin(x, y):
    xx = np.asarray(x, float); yy = np.asarray(y, float)
    denom = np.dot(xx, xx)
    if denom == 0: return None
    return float(np.dot(xx, yy) / denom)

def r2_linear(y_true, y_fit):
    y = np.asarray(y_true, float); yf = np.asarray(y_fit, float)
    ss_res = np.sum((y - yf) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def find_rotation_at_mrd(rot, mom, mrd):
    s = mom - mrd
    hits = np.where(np.isclose(s, 0.0, atol=1e-12))[0]
    if hits.size: return float(rot[hits[-1]])
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0: return None
    i = int(idx[-1]); x0, x1 = rot[i], rot[i+1]; y0, y1 = mom[i], mom[i+1]
    return float(x0 + (mrd - y0) * (x1 - x0) / (y1 - y0))

def window_slope_by_percent(rot, mom, p, theta_min=1e-9):
    Mmax = float(np.max(mom))
    lim  = (p / 100.0) * Mmax
    mask = (mom <= lim) & (rot > theta_min)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2: return None, None, mask
    k = ols_slope_through_origin(xx, yy)
    if k is None or not np.isfinite(k): return None, None, mask
    r2 = r2_linear(yy, k * xx)
    return k, r2, mask

def choose_auto_window(rot, mom,
                       r2_strict=0.998, r2_loose=0.995,
                       sens_strict=0.03, sens_loose=0.05):
    """
    Initial-slope biased auto-window:
      - p-grid dense near zero: 0.5, 0.75, 1.0, 1.5, 2..12 %
      - Adaptive min points: ~10% of N, clamped [4, 10]
      - Prefer the SMALLEST p that passes:
          * n >= min_pts
          * R² >= r2_strict (else >= r2_loose)
          * slope stability vs immediately smaller p: Δk/k <= sens_strict (else <= sens_loose)
      - Fallback: smallest p with highest R² (n >= 3)
    """
    N = len(rot)
    min_pts = int(max(4, min(10, round(0.10 * N))))
    p_grid = [0.5, 0.75, 1.0, 1.5] + list(range(2, 13))  # 0.5–12%

    prev = None
    strict_ok = None
    loose_ok  = None
    best_r2   = None  # for fallback

    results = []
    for p in p_grid:
        k, r2, mask = window_slope_by_percent(rot, mom, p, theta_min=1e-9)
        if k is None: 
            results.append((p, None, None, 0, mask))
            continue
        n = int(mask.sum())
        results.append((p, k, r2, n, mask))

        # check stability against immediately smaller successful window
        sens = None
        if prev and prev["k"] is not None and prev["n"] >= 2 and k != 0:
            sens = abs(k - prev["k"]) / abs(prev["k"])

        # Track best R² for fallback (prefer smaller p on ties)
        if n >= 3 and (best_r2 is None or r2 > best_r2["r2"] or (r2 == best_r2["r2"] and p < best_r2["p"])):
            best_r2 = {"p": p, "k": k, "r2": r2, "n": n, "mask": mask, "reason": "auto (best R² fallback)"}

        # Strict acceptance first
        if n >= min_pts and r2 is not None:
            if sens is not None and sens <= sens_strict and r2 >= r2_strict and strict_ok is None:
                strict_ok = {"p": p, "k": k, "r2": r2, "n": n, "mask": mask, "reason": "auto (strict)"}
            elif sens is not None and sens <= sens_loose and r2 >= r2_loose and loose_ok is None:
                loose_ok = {"p": p, "k": k, "r2": r2, "n": n, "mask": mask, "reason": "auto (loose)"}

        # update prev only if this window had a valid slope
        prev = {"p": p, "k": k, "r2": r2, "n": n}

    if strict_ok is not None:
        return strict_ok
    if loose_ok is not None:
        return loose_ok
    return best_r2

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

# ---------- Load data (assume exactly 2 columns: Rotation, Moment) ----------
df = pd.read_excel(uploaded)
x = df.iloc[:, 0].to_numpy(float)
y = df.iloc[:, 1].to_numpy(float)

# Tidy: drop NaNs, sort by rotation
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]; y = y[mask]
order = np.argsort(x)
x = x[order]; y = y[order]

# ---------- Compute Sj,ini ----------
if mode == "Auto (recommended)":
    choice = choose_auto_window(x, y)
    if choice is None:
        st.error("Could not determine a stable initial window. Try Manual p% (e.g., 1–3%).")
        st.stop()
    Sj_ini = choice["k"]; used_r2 = choice["r2"]; used_mask = choice["mask"]; note = f"{choice['reason']}, p≤{choice['p']}%"
else:
    k, r2, mask_p = window_slope_by_percent(x, y, p_manual, theta_min=1e-9)
    if k is None:
        st.error("Not enough points in the selected window. Increase p% slightly.")
        st.stop()
    Sj_ini = k; used_r2 = r2; used_mask = mask_p; note = f"manual, p≤{p_manual:g}%"

Sj = Sj_ini / 2.0

# ---------- Mrd & stiffness lines ----------
x_mrd = find_rotation_at_mrd(x, y, mrd)
x_max = float(np.max(x)) if x.size else 0.0
x_end_ini = (mrd / Sj_ini) if Sj_ini != 0 else x_max
x_end_sj  = (mrd / Sj)     if Sj     != 0 else x_max
limits = [v for v in [x_end_ini, x_end_sj, (x_mrd if x_mrd is not None else x_max), x_max] if np.isfinite(v)]
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
