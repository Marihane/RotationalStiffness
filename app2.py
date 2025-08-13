# app.py
# Rotational stiffness interactive app (robust Sj,ini with auto window)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rotational Stiffness", layout="wide")
st.title("Rotational stiffness")
st.caption("Upload an Excel file with columns Rotation and Moment")

# ----------------- Helpers -----------------
def rot_at_mrd(rot, mom, target):
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

def ols_through_origin(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    denom = float(np.dot(x, x))
    if denom == 0: return None
    return float(np.dot(x, y) / denom)

def r2_with_intercept(y_true, y_pred):
    # Standard R² (w.r.t. mean of y_true) to detect curvature/nonlinearity
    y = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def slope_on_percent_window(rot, mom, p_percent, min_rot=0.0):
    Mmax = float(np.nanmax(mom)) if mom.size else 0.0
    lim = (p_percent / 100.0) * Mmax
    mask = (mom <= lim) & (rot > min_rot) & np.isfinite(rot) & np.isfinite(mom)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2:
        return None, None, None, mask
    k = ols_through_origin(xx, yy)
    if k is None or not np.isfinite(k):
        return None, None, None, mask
    r2 = r2_with_intercept(yy, k * xx)
    return k, r2, lim, mask

def auto_choose_window(rot, mom, p_min=5, p_max=20, r2_target=0.995, min_pts=8, sens_tol=0.05):
    """
    Sweep p in [p_min, p_max] (% of peak M). Prefer the LARGEST p that:
      - has ≥ min_pts points,
      - R² ≥ r2_target,
      - sensitivity (vs p/2) < sens_tol.
    If none pass all checks, return the best available (highest R² with enough points),
    otherwise fall back to smaller constraints.
    """
    candidates = []
    for p in range(int(p_min), int(p_max) + 1):
        k, r2, lim, mask = slope_on_percent_window(rot, mom, p)
        n = int(mask.sum()) if mask is not None else 0
        if k is None: continue
        candidates.append({"p": p, "k": k, "r2": r2, "n": n, "mask": mask})

    if not candidates:
        return None  # caller will handle fallback

    # Compute sensitivity: compare slope at p to slope at p/2 (nearest available)
    p_values = [c["p"] for c in candidates]
    p_to_k = {c["p"]: c["k"] for c in candidates}

    def sens_for(c):
        p = c["p"]; p_half = max(min(p_values), int(round(p / 2)))
        k_half = p_to_k.get(p_half, None)
        if k_half is None or k_half == 0: return np.inf
        return abs(c["k"] - k_half) / abs(k_half)

    for c in candidates:
        c["sens"] = sens_for(c)

    # First pass: meet all criteria, pick largest p
    ok = [c for c in candidates if c["n"] >= min_pts and c["r2"] >= r2_target and c["sens"] <= sens_tol]
    if ok:
        best = max(ok, key=lambda c: c["p"])
        best["reason"] = "meets R², min_pts, sensitivity"
        return best

    # Second pass: relax sensitivity, keep R² + min_pts, pick largest p
    ok = [c for c in candidates if c["n"] >= min_pts and c["r2"] >= r2_target]
    if ok:
        best = max(ok, key=lambda c: c["p"])
        best["reason"] = "meets R² and min_pts"
        return best

    # Third pass: pick the one with max R² (still require at least 3 points)
    ok = [c for c in candidates if c["n"] >= max(3, min_pts//2)]
    if ok:
        best = max(ok, key=lambda c: c["r2"])
        best["reason"] = "best available R²"
        return best

    # Last resort: absolute max R²
    best = max(candidates, key=lambda c: c["r2"])
    best["reason"] = "last resort"
    return best

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Inputs")
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=1590.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness - Example")

    st.markdown("---")
    st.subheader("Initial stiffness window (auto)")
    p_min = st.slider("Min % of peak M", 1, 30, 5, 1)
    p_max = st.slider("Max % of peak M", p_min, 40, 15, 1)
    r2_target = st.number_input("Target R² (linearity)", min_value=0.90, max_value=1.00, value=0.995, step=0.001, format="%.3f")
    min_pts = st.number_input("Minimum points in window", min_value=3, value=8, step=1)
    sens_tol = st.number_input("Sensitivity tolerance (Δk/k at p vs p/2)", min_value=0.0, max_value=0.50, value=0.05, step=0.01)

    st.markdown("---")
    st.subheader("Fallback controls")
    k_fallback = st.number_input("K points (derivative fallback)", min_value=2, value=3, step=1)
    theta_min = st.number_input("Skip rotations < θ_min [rad]", min_value=0.0, value=0.0, step=1e-6, format="%.6f")

uploaded = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    help="The first sheet should contain columns Rotation and Moment"
)

if not uploaded:
    st.info("Upload an Excel file to begin")
    st.stop()

# ----------------- Load & prep data -----------------
try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

cols = list(df.columns)
c1, c2 = st.columns(2)
with c1:
    rot_col = st.selectbox("Rotation column", cols, index=cols.index("Rotation") if "Rotation" in cols else 0)
with c2:
    mom_col = st.selectbox("Moment column", cols, index=cols.index("Moment") if "Moment" in cols else 1)

try:
    rotation = df[rot_col].to_numpy(dtype=float)
    moment = df[mom_col].to_numpy(dtype=float)
except Exception as e:
    st.error(f"Could not parse numeric data: {e}")
    st.stop()

mask_valid = np.isfinite(rotation) & np.isfinite(moment)
rotation = rotation[mask_valid]
moment = moment[mask_valid]

if rotation.size < 2:
    st.error("Need at least two valid data points")
    st.stop()

# Sort by rotation
ordr = np.argsort(rotation)
rotation = rotation[ordr]
moment = moment[ordr]

# ----------------- Compute Sj,ini (auto) -----------------
choice = auto_choose_window(rotation, moment, p_min=p_min, p_max=p_max,
                            r2_target=r2_target, min_pts=int(min_pts), sens_tol=float(sens_tol))

method_note = ""
win_mask = np.zeros_like(rotation, dtype=bool)
Sj_ini = None
used_p = None
used_r2 = None
used_n = None

if choice is not None:
    Sj_ini = choice["k"]; used_p = choice["p"]; used_r2 = choice["r2"]; used_n = choice["n"]
    win_mask = choice["mask"]; method_note = f"(auto window p ≤ {used_p}% of peak; {choice['reason']})"

# Fallback: local derivative near origin (OLS through origin on first K positive points)
if Sj_ini is None or not np.isfinite(Sj_ini):
    mask_pos = (rotation > max(theta_min, 0.0))
    xx = rotation[mask_pos][:int(k_fallback)]
    yy = moment[mask_pos][:int(k_fallback)]
    if xx.size >= 2:
        Sj_ini = ols_through_origin(xx, yy)
        used_p = None
        used_r2 = r2_with_intercept(yy, Sj_ini * xx) if np.isfinite(Sj_ini) else None
        used_n = xx.size
        win_mask = np.zeros_like(rotation, dtype=bool)
        win_mask[:used_n] = True
        method_note = f"(fallback: first {used_n} positive-rotation points)"
    else:
        st.error("Not enough early points to estimate Sj,ini.")
        st.stop()

Sj = Sj_ini / 2.0

# ----------------- Mrd + stiffness lines -----------------
x_mrd = rot_at_mrd(rotation, moment, mrd)
x_max = float(np.nanmax(rotation))
x_lim_ini = (mrd / Sj_ini) if (Sj_ini not in (None, 0)) else None
x_lim_sj  = (mrd / Sj)     if (Sj     not in (None, 0)) else None
x_limit = safe_min(x_lim_ini, x_lim_sj, (x_mrd if x_mrd is not None else x_max), x_max)

x_line = np.array([0.0, x_limit]) if (x_limit is not None and np.isfinite(x_limit)) else np.array([0.0, x_max])
y_line_ini = Sj_ini * x_line
y_line_sj  = Sj * x_line

# ----------------- Plot -----------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(rotation, moment, label="Moment–Rotation", color="black", linewidth=1.8)

# Highlight window points used for Sj,ini
if win_mask.any():
    ax.scatter(rotation[win_mask], moment[win_mask], s=28, edgecolor="blue", facecolor="none", linewidth=1.2, label="Points used for Sj,ini")

ax.plot(x_line, y_line_ini, "--", label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad {method_note}", color="blue")
ax.plot(x_line, y_line_sj,  "--", label=f"Sj ≈ {Sj:.1f} kNm/rad", color="green")
ax.axhline(mrd, linestyle=":", color="red", label=f"Mrd = {mrd:.1f} kNm")
if x_mrd is not None and np.isfinite(x_mrd):
    ax.plot(x_mrd, mrd, "ro", label="Mrd point")
else:
    ax.text(0.02, 0.95, "Warning: Mrd does not intersect within the curve range",
            transform=ax.transAxes, fontsize=9, color="red", va="top")

ax.set_xlabel("Rotation [rad]")
ax.set_ylabel("Moment [kNm]")
ax.set_title(title)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend()
fig.tight_layout()
st.pyplot(fig, clear_figure=False)

# ----------------- Results -----------------
st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sj,ini [kNm/rad]", f"{Sj_ini:.1f}")
c2.metric("Sj [kNm/rad]", f"{Sj:.1f}")
if x_mrd is not None and np.isfinite(x_mrd):
    c3.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}")
else:
    c3.metric("Rotation at Mrd [rad]", "no intersection")
c4.metric("R² of window", f"{used_r2:.5f}" if used_r2 is not None else "n/a")

st.caption(
    f"Window: {'auto' if used_p is not None else 'fallback'}, "
    f"{'p≤'+str(used_p)+'%' if used_p is not None else ''} "
    f"({used_n} points)."
)
