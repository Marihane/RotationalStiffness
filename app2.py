# app.py
# Rotational stiffness interactive app

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Rotational Stiffness", layout="wide")

st.title("Rotational stiffness")
st.caption("Upload an Excel file with columns Rotation and Moment")

# ---------- Helpers ----------
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


def ols_through_origin(x, y):
    """Slope k minimizing ||y - kx||_2 subject to intercept=0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.dot(x, x)
    if denom == 0:
        return None
    return float(np.dot(x, y) / denom)


def finite_diff_initial_slope(x, y, k_points=3):
    """Finite-difference slope near origin using first k_points with x>0.
    Uses a simple linear fit through origin on those points as a stable estimate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    xx = x[mask][:k_points]
    yy = y[mask][:k_points]
    if xx.size < 2:
        return None
    return ols_through_origin(xx, yy)


# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Inputs")
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=1590.0, step=10.0)
    i_tangent = st.number_input("I_TANGENT (index)", min_value=1, value=3, step=1)
    title = st.text_input("Plot title", "Rotational Stiffness - Example")

    st.markdown("---")
    st.subheader("Sj,ini method")
    sj_method = st.radio(
        "How to compute Sj,ini",
        options=[
            "By index (origin→point)",
            "OLS (first N points)",
            "OLS (≤ p% of peak M)",
            "Local derivative near origin",
        ],
        index=2,  # default to robust % of peak
    )

    colA, colB = st.columns(2)
    with colA:
        n_first = st.number_input("N (for 'first N')", min_value=2, value=8, step=1)
    with colB:
        p_peak = st.slider("p% of peak M (for '≤ p%')", min_value=1, max_value=40, value=10, step=1)

    k_points = st.number_input("K points (for derivative)", min_value=2, value=3, step=1)


uploaded = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    help="The first sheet should contain columns Rotation and Moment"
)

if not uploaded:
    st.info("Upload an Excel file to begin")
    st.stop()

# ---------- Read Excel ----------
try:
    df = pd.read_excel(uploaded)  # uses openpyxl
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

# Let the user map columns if names differ
cols = list(df.columns)
c1, c2 = st.columns(2)
with c1:
    rot_col = st.selectbox("Rotation column", cols, index=cols.index("Rotation") if "Rotation" in cols else 0)
with c2:
    mom_col = st.selectbox("Moment column", cols, index=cols.index("Moment") if "Moment" in cols else 1)

# Extract arrays
try:
    rotation = df[rot_col].to_numpy(dtype=float)
    moment = df[mom_col].to_numpy(dtype=float)
except Exception as e:
    st.error(f"Could not parse numeric data: {e}")
    st.stop()

# Basic checks
if rotation.size < 2 or moment.size < 2:
    st.error("Need at least two data points")
    st.stop()
if not (1 <= i_tangent < rotation.size):
    st.error(f"I_TANGENT must be between 1 and {rotation.size - 1}")
    st.stop()

# ---------- Clean data for fitting ----------
mask_valid = np.isfinite(rotation) & np.isfinite(moment)
rotation = rotation[mask_valid]
moment = moment[mask_valid]

# Sort by rotation just in case
order = np.argsort(rotation)
rotation = rotation[order]
moment = moment[order]

# ---------- Compute Sj,ini by selected method ----------
Sj_ini = None
method_note = ""

if sj_method == "By index (origin→point)":
    x_tan = rotation[i_tangent]
    y_tan = moment[i_tangent]
    if x_tan == 0:
        st.error("Rotation at I_TANGENT is zero so Sj,ini is undefined")
        st.stop()
    Sj_ini = y_tan / x_tan
    method_note = f"(origin→point at index {i_tangent})"

elif sj_method == "OLS (first N points)":
    n = int(min(n_first, rotation.size))
    xx = rotation[:n]
    yy = moment[:n]
    # Drop zeros to avoid bias if first rows are exactly 0 rotation
    fit_mask = xx > 0
    Sj_ini = ols_through_origin(xx[fit_mask], yy[fit_mask])
    method_note = f"(OLS through origin using first {n} points)"

elif sj_method == "OLS (≤ p% of peak M)":
    Mmax = float(np.nanmax(moment)) if moment.size else 0.0
    Mlim = (p_peak / 100.0) * Mmax
    fit_mask = (moment <= Mlim) & (rotation > 0)
    xx = rotation[fit_mask]
    yy = moment[fit_mask]
    if xx.size < 2:  # fallback: first N
        n = int(min(max(5, n_first), rotation.size))
        xx = rotation[:n]
        yy = moment[:n]
        fit_mask2 = xx > 0
        Sj_ini = ols_through_origin(xx[fit_mask2], yy[fit_mask2])
        method_note = f"(OLS fallback on first {n} points; not enough data ≤ {p_peak}% of peak)"
    else:
        Sj_ini = ols_through_origin(xx, yy)
        method_note = f"(OLS through origin using points with M ≤ {p_peak}% of peak)"

elif sj_method == "Local derivative near origin":
    Sj_ini = finite_diff_initial_slope(rotation, moment, k_points=int(k_points))
    method_note = f"(finite-difference using first {int(k_points)} positive-rotation points)"

if Sj_ini is None or not np.isfinite(Sj_ini):
    st.error("Could not determine Sj,ini with the selected method/data.")
    st.stop()

Sj = Sj_ini / 2.0

# ---------- Mrd intersection and stiffness line limits ----------
x_mrd = rot_at_mrd(rotation, moment, mrd)
x_max = float(np.nanmax(rotation))
x_lim_ini = (mrd / Sj_ini) if (Sj_ini not in (None, 0)) else None
x_lim_sj = (mrd / Sj) if (Sj not in (None, 0)) else None
x_limit = safe_min(x_lim_ini, x_lim_sj, (x_mrd if x_mrd is not None else x_max), x_max)

x_line = np.array([0.0, x_limit]) if (x_limit is not None and np.isfinite(x_limit)) else np.array([0.0, x_max])
y_line_ini = Sj_ini * x_line
y_line_sj = Sj * x_line

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(rotation, moment, label="Moment–Rotation", color="black", linewidth=1.8)
ax.plot(x_line, y_line_ini, "--", label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad {method_note}", color="blue")
ax.plot(x_line, y_line_sj, "--", label=f"Sj ≈ {Sj:.1f} kNm/rad", color="green")
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

# ---------- Results panel ----------
st.subheader("Results")
c3, c4, c5 = st.columns(3)
c3.metric("Sj,ini [kNm/rad]", f"{Sj_ini:.1f}")
c4.metric("Sj [kNm/rad]", f"{Sj:.1f}")
if x_mrd is not None and np.isfinite(x_mrd):
    c5.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}")
else:
    c5.metric("Rotation at Mrd [rad]", "no intersection")
