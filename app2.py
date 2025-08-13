# app.py — Rotational stiffness (simple, robust Sj,ini)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Rotational Stiffness", layout="wide")
st.title("Rotational stiffness")
st.caption("Upload an Excel file with columns Rotation and Moment")

# ---------- Helpers ----------
def rot_at_mrd(rot, mom, target):
    rot = np.asarray(rot, dtype=float); mom = np.asarray(mom, dtype=float)
    hits = np.where(np.isclose(mom, target, atol=1e-9))[0]
    if hits.size: return float(rot[hits[-1]])
    s = mom - target
    idx = np.where((s[:-1] == 0) | (s[:-1] * s[1:] < 0) | (s[1:] == 0))[0]
    if idx.size == 0: return None
    i = idx[-1]; x0, x1 = rot[i], rot[i+1]; y0, y1 = mom[i], mom[i+1]
    if y1 == y0: return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

def safe_min(*vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return min(vals) if vals else None

def ols_through_origin(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    denom = float(np.dot(x, x))
    if denom == 0: return None
    return float(np.dot(x, y) / denom)

def r2_with_intercept(y_true, y_pred):
    y = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y - yp)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0

def slope_on_percent_window(rot, mom, p_percent, theta_min=1e-9):
    Mmax = float(np.nanmax(mom)) if mom.size else 0.0
    lim = (p_percent/100.0) * Mmax
    mask = (mom <= lim) & (rot > theta_min) & np.isfinite(rot) & np.isfinite(mom)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2: return None, None, None, mask
    k = ols_through_origin(xx, yy)
    if k is None or not np.isfinite(k): return None, None, None, mask
    r2 = r2_with_intercept(yy, k*xx)
    return k, r2, lim, mask

def auto_choose_window(rot, mom, p_min=5, p_max=15, r2_target=0.995, min_pts=8, sens_tol=0.05):
    """Pick the largest p in [p_min, p_max] that is linear & stable."""
    candidates = []
    for p in range(int(p_min), int(p_max)+1):
        k, r2, lim, mask = slope_on_percent_window(rot, mom, p)
        if k is None: continue
        candidates.append({"p": p, "k": k, "r2": r2, "n": int(mask.sum()), "mask": mask})

    if not candidates: return None

    # Sensitivity check: compare slope at p vs approx p/2
    p_to_k = {c["p"]: c["k"] for c in candidates}
    p_vals = [c["p"] for c in candidates]

    def sens(c):
        p = c["p"]; half = max(min(p_vals), int(round(p/2)))
        k_half = p_to_k.get(half)
        if not k_half: return np.inf
        return abs(c["k"] - k_half)/abs(k_half)

    for c in candidates: c["sens"] = sens(c)

    # Pass 1: all criteria
    ok = [c for c in candidates if c["n"] >= min_pts and c["r2"] >= r2_target and c["sens"] <= sens_tol]
    if ok: 
        best = max(ok, key=lambda c: c["p"]); best["reason"]="auto: linear & stable"; return best

    # Pass 2: relax sensitivity
    ok = [c for c in candidates if c["n"] >= min_pts and c["r2"] >= r2_target]
    if ok: 
        best = max(ok, key=lambda c: c["p"]); best["reason"]="auto: linear"; return best

    # Fallback: best R² with ≥3 points
    ok = [c for c in candidates if c["n"] >= 3]
    best = max(ok, key=lambda c: c["r2"]) if ok else max(candidates, key=lambda c: c["r2"])
    best["reason"]="auto: best available"
    return best

# ---------- Sidebar (minimal) ----------
with st.sidebar:
    st.header("Inputs")
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=1590.0, step=10.0)
    title = st.text_input("Plot title", "Rotational Stiffness - Example")

    st.markdown("---")
    mode = st.radio("Initial stiffness window", ["Auto (recommended)", "Manual p%"], index=0)
    if mode == "Manual p%":
        p_manual = st.slider("p% of peak M (window size)", 5, 20, 10, 1)

uploaded = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    help="The first sheet should contain columns Rotation and Moment"
)

if not uploaded:
    st.info("Upload an Excel file to begin")
    st.stop()

# ---------- Load & prep ----------
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
rotation = rotation[mask_valid]; moment = moment[mask_valid]
if rotation.size < 2: 
    st.error("Need at least two valid data points"); st.stop()

order = np.argsort(rotation)
rotation = rotation[order]; moment = moment[order]

# ---------- Compute Sj,ini ----------
theta_min = 1e-9  # ignore exact zeros / backlash numerics
Sj_ini = None; used_r2=None; used_p=None; win_mask = np.zeros_like(rotation, bool); note=""

if mode == "Auto (recommended)":
    choice = auto_choose_window(rotation, moment, p_min=5, p_max=15, r2_target=0.995, min_pts=8, sens_tol=0.05)
    if choice is not None:
        Sj_ini = choice["k"]; used_r2 = choice["r2"]; used_p = choice["p"]; win_mask = choice["mask"]; note = choice["reason"]

if (Sj_ini is None) and (mode == "Manual p%"):
    k, r2, _, mask = slope_on_percent_window(rotation, moment, p_manual, theta_min=theta_min)
    if k is not None:
        Sj_ini = k; used_r2 = r2; used_p = p_manual; win_mask = mask; note = "manual window"

# Last resort fallback: first 3 positive-rotation points
if Sj_ini is None or not np.isfinite(Sj_ini):
    pos = rotation > theta_min
    xx = rotation[pos][:3]; yy = moment[pos][:3]
    if xx.size >= 2:
        Sj_ini = ols_through_origin(xx, yy)
        used_r2 = r2_with_intercept(yy, Sj_ini*xx)
        used_p = None
        win_mask = np.zeros_like(rotation, bool); win_mask[:xx.size] = True
        note = "fallback: first 2–3 points"
    else:
        st.error("Not enough early points to estimate Sj,ini."); st.stop()

Sj = Sj_ini / 2.0

# ---------- Mrd & lines ----------
x_mrd = rot_at_mrd(rotation, moment, mrd)
x_max = float(np.nanmax(rotation))
x_lim_ini = (mrd / Sj_ini) if Sj_ini else None
x_lim_sj  = (mrd / Sj)     if Sj     else None
x_limit = safe_min(x_lim_ini, x_lim_sj, x_mrd if x_mrd is not None else x_max, x_max)

x_line = np.array([0.0, x_limit]) if (x_limit is not None and np.isfinite(x_limit)) else np.array([0.0, x_max])
y_line_ini = Sj_ini * x_line
y_line_sj  = Sj * x_line

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(rotation, moment, label="Moment–Rotation", color="black", linewidth=1.8)

# show points used for Sj,ini
if win_mask.any():
    ax.scatter(rotation[win_mask], moment[win_mask], s=28, edgecolor="blue", facecolor="none",
               linewidth=1.2, label="Points used for Sj,ini")

ax.plot(x_line, y_line_ini, "--", label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad ({note}{'' if used_p is None else f', p≤{used_p}%'} )", color="blue")
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

# ---------- Results ----------
st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sj,ini [kNm/rad]", f"{Sj_ini:.1f}")
c2.metric("Sj [kNm/rad]", f"{Sj:.1f}")
c3.metric("R² (window)", f"{used_r2:.5f}" if used_r2 is not None else "n/a")
if x_mrd is not None and np.isfinite(x_mrd):
    c4.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}")
else:
    c4.metric("Rotation at Mrd [rad]", "no intersection")
