# app.py — Rotational stiffness (simple + sturdy)
# Auto = steepest initial prefix; Manual = first K points (always works)

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
- **Manual (first K points):** uses exactly the **first K** points by rotation (K≥2).
  If K is too small/large for the data, the app automatically adjusts so the fit stays valid.
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
    Track the running max slope; return the steepest straight prefix
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

def manual_first_k_always(rot, mom, K, theta_min=1e-9):
    """
    Manual: use first K positive-rotation points. If that yields a bad/degenerate
    fit, automatically adjust (increase K up to available; if still no good,
    fall back to the first two points overall). Returns (k, r2, mask, note).
    """
    n = rot.size
    if n < 2:
        return None, None, None, "not enough data"

    pos_idx = np.where(rot > theta_min)[0]
    # preferred set: first positive-rotation points
    if pos_idx.size >= 2:
        maxK = int(pos_idx.size)
        K_use = int(max(2, min(K, maxK)))
        # try increasing K until slope is valid
        for kk in range(K_use, maxK + 1):
            idx = pos_idx[:kk]
            xx, yy = rot[idx], mom[idx]
            k_slope = ols_slope_through_origin(xx, yy)
            if k_slope is not None and np.isfinite(k_slope):
                r2 = r2_linear(yy, k_slope * xx)
                mask = np.zeros_like(rot, dtype=bool); mask[idx] = True
                note = f"manual K (first {kk} points)"
                if kk != K:
                    note += f" (auto-adjusted from K={K})"
                return k_slope, r2, mask, note

    # fallback: take the first two points overall by rotation
    order = np.argsort(rot)
    idx2 = order[: min(2, n)]
    if idx2.size == 2:
        xx, yy = rot[idx2], mom[idx2]
        k_slope = ols_slope_through_origin(xx, yy)
        if k_slope is not None and np.isfinite(k_slope):
            r2 = r2_linear(yy, k_slope * xx)
            mask = np.zeros_like(rot, dtype=bool); mask[idx2] = True
            return k_slope, r2, mask, "manual K fallback (first 2 overall)"
    return None, None, None, "manual K: could not form a valid window"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=340.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness")
    mode = st.radio("Initial stiffness window", ["Auto (recommended)", "Manual (first K points)"], index=0)
    if mode == "Manual (first K points)":
        K_manual = st.slider("K (first points to use)", 2, 30, 2, 1)

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

# ── Compute Sj,ini ────────────────────────────────────────────────────────────
if mode == "Auto (recommended)":
    choice = choose_initial_prefix(x, y, k_cap=30, r2_min=0.995, drop_tol=0.05, theta_min=theta_min)
    if choice is None:
        # last-resort: first 2 positive-rotation points
        Sj_ini, used_r2, used_mask, note = manual_first_k_always(x, y, K=2, theta_min=theta_min)
        if Sj_ini is None:
            st.error("Not enough early points to estimate Sj,ini.")
            st.stop()
        note = "auto fallback → " + note
    else:
        Sj_ini = choice["slope"]; used_r2 = choice["r2"]; used_mask = choice["mask"]; note = choice["reason"]
else:
    Sj_ini, used_r2, used_mask, note = manual_first_k_always(x, y, K_manual, theta_min=theta_min)
    if Sj_ini is None:
        st.error("Manual K: could not form a valid window.")
        st.stop()

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
