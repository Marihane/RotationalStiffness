# app.py — Rotational stiffness (Auto + Manual sliding window)
# Auto = steepest initial prefix
# Manual = consecutive K-point window with a movable start index

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rotational Stiffness", layout="wide")
st.title("Rotational stiffness")
st.caption("Upload an Excel file with two columns: Rotation (x) and Moment (y).")

# ── Tiny explainer ────────────────────────────────────────────────────────────
with st.expander("What the app does (short)", expanded=False):
    st.markdown(
        """
**Goal:** estimate the **initial rotational stiffness** \(S_{j,ini}\) — the slope near the origin.

- **Auto (recommended):** looks at the first few points (smallest rotations), fits lines through the origin,
  and picks the **steepest straight prefix** before the curve begins to bend.
- **Manual (sliding window):** choose a **window size K** (number of consecutive points) and **move it**
  along the early part of the curve with a **Start at point j** slider. The slope is an OLS line **through the origin**
  using only those K points.
        """
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def ols_slope_through_origin(x, y):
    """Return slope k minimizing sum (y - kx)^2 with intercept = 0."""
    xx = np.asarray(x, float); yy = np.asarray(y, float)
    denom = float(np.dot(xx, xx))
    if denom == 0:
        return None
    return float(np.dot(xx, yy) / denom)

def r2_linear(y_true, y_fit):
    y = np.asarray(y_true, float); yf = np.asarray(y_fit, float)
    ss_res = float(np.sum((y - yf) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0

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

def choose_initial_prefix(rot, mom, k_cap=30, r2_min=0.995, drop_tol=0.05, theta_min=1e-9):
    """
    Auto: scan prefixes k=2..Kcap of the first positive-rotation points.
    Track running max slope; return the steepest straight prefix
    before a significant slope drop (> drop_tol).
    """
    pos = rot > theta_min
    X = rot[pos]; Y = mom[pos]
    if X.size < 2:
        return None  # not enough early data

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
                    return {"slope": running_max, "r2": running_r2, "mask": running_mask,
                            "reason": f"auto prefix: steepest before bend (k={running_k})"}
                if best is not None:
                    return {"slope": best, "r2": best_r2, "mask": best_mask,
                            "reason": f"auto prefix: best R² & steep (k={best_k})"}

    if best is not None:
        return {"slope": best, "r2": best_r2, "mask": best_mask,
                "reason": f"auto prefix: best R² & steep (k={best_k})"}
    if running_k is not None:
        return {"slope": running_max, "r2": running_r2, "mask": running_mask,
                "reason": f"auto prefix: running max (k={running_k})"}
    return None

def select_window_by_offset(rot, mom, start_j, K, theta_min=1e-9):
    """
    Manual sliding window: take K consecutive **positive-rotation** points,
    starting at the j-th positive-rotation point (1-based).
    Always returns a valid window (auto-clamps j and K).
    """
    pos_idx = np.where(rot > theta_min)[0]
    npos = int(pos_idx.size)
    if npos < 2:
        return None, None, None, "manual window: not enough early points"

    # Clamp K and start_j to valid ranges
    K = int(max(2, min(K, npos)))
    max_start = max(1, npos - K + 1)
    start_j = int(max(1, min(start_j, max_start)))

    sel = pos_idx[start_j - 1 : start_j - 1 + K]
    xx, yy = rot[sel], mom[sel]

    k_slope = ols_slope_through_origin(xx, yy)
    if k_slope is None or not np.isfinite(k_slope):
        # very unlikely with clamped ranges; fallback to first 2
        sel = pos_idx[:2]; xx, yy = rot[sel], mom[sel]
        k_slope = ols_slope_through_origin(xx, yy)
        note = f"manual window auto-fallback (first 2 points)"
    else:
        note = f"manual window: K={K}, start j={start_j}"

    r2 = r2_linear(yy, k_slope * xx)
    mask = np.zeros_like(rot, dtype=bool); mask[sel] = True
    return k_slope, r2, mask, note, npos, K, start_j

# ── Sidebar: only stable controls up-front ────────────────────────────────────
with st.sidebar:
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=340.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness")
    mode = st.radio("Initial stiffness window", ["Auto (recommended)", "Manual (sliding window)"], index=0)

# ── File upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    help="Two columns: Rotation (x), Moment (y). First row can be header."
)
if not uploaded:
    st.info("Upload an Excel file to begin")
    st.stop()

# ── Load & tidy (assume exactly 2 columns: Rotation, Moment) ──────────────────
df = pd.read_excel(uploaded)
x = df.iloc[:, 0].to_numpy(float)
y = df.iloc[:, 1].to_numpy(float)
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]; y = y[mask]
order = np.argsort(x)
x = x[order]; y = y[order]
theta_min = 1e-9

# ── If Manual mode, show the dynamic sliders now that we know data size ───────
if mode == "Manual (sliding window)":
    pos_idx = np.where(x > theta_min)[0]
    npos = int(pos_idx.size)
    if npos < 2:
        st.error("Not enough early (positive rotation) points for a manual window.")
        st.stop()
    max_K = int(min(30, npos))
    # Defaults: K=2, start at 1
    with st.sidebar:
        K_manual = st.slider("K (consecutive points)", 2, max_K, 2, 1)
        max_start = max(1, npos - K_manual + 1)
        start_j = st.slider("Start at point j (among early points)", 1, max_start, 1, 1)
else:
    K_manual = None
    start_j = None

# ── Compute Sj,ini ────────────────────────────────────────────────────────────
if mode == "Auto (recommended)":
    choice = choose_initial_prefix(x, y, k_cap=30, r2_min=0.995, drop_tol=0.05, theta_min=theta_min)
    if choice is None:
        # last-resort: first 2 positive-rotation points
        k_slope, used_r2, used_mask, note, *_ = select_window_by_offset(x, y, start_j=1, K=2, theta_min=theta_min)
        if k_slope is None:
            st.error("Not enough data to estimate Sj,ini.")
            st.stop()
        Sj_ini = k_slope; note = "auto fallback → first 2 points"
    else:
        Sj_ini = choice["slope"]; used_r2 = choice["r2"]; used_mask = choice["mask"]; note = choice["reason"]

else:  # Manual sliding window
    k_slope, used_r2, used_mask, note, npos, K_final, j_final = select_window_by_offset(
        x, y, start_j=start_j, K=K_manual, theta_min=theta_min
    )
    if k_slope is None:
        st.error("Manual window: could not form a valid window.")
        st.stop()
    Sj_ini = k_slope
    # reflect any auto clamping in the note
    note = f"{note} (npos={npos})"

Sj = Sj_ini / 2.0

# ── Mrd & stiffness lines ─────────────────────────────────────────────────────
x_mrd = find_rotation_at_mrd(x, y, mrd)
x_max = float(np.max(x)) if x.size else 0.0
x_end_ini = (mrd / Sj_ini) if Sj_ini else x_max
x_end_sj  = (mrd / Sj)     if Sj     else x_max
vals = [v for v in [x_end_ini, x_end_sj, (x_mrd if x_mrd is not None else x_max), x_max] if v is not None and np.isfinite(v)]
x_limit = float(np.min(vals)) if vals else x_max

x_line = np.array([0.0, x_limit], float)
y_line_ini = Sj_ini * x_line
y_line_sj  = Sj * x_line

# ── Plot ──────────────────────────────────────────────────────────────────────
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

ax.set_xlabel("Rotation [rad]")
ax.set_ylabel("Moment [kNm]")
ax.set_title(title)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend()
fig.tight_layout()
st.pyplot(fig, clear_figure=False)

# ── Results ───────────────────────────────────────────────────────────────────
st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sj,ini [kNm/rad]", f"{Sj_ini:.1f}")
c2.metric("Sj [kNm/rad]", f"{Sj:.1f}")
c3.metric("R² (window)", f"{used_r2:.5f}" if used_r2 is not None else "n/a")
c4.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}" if x_mrd is not None else "no intersection")
