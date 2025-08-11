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

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    mrd = st.number_input("Mrd [kNm]", min_value=0.0, value=1590.0, step=10.0)
    i_tangent = st.number_input("Index", min_value=1, value=3, step=1)
    title = st.text_input("Plot title", "Rotational Stiffness")


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


uploaded = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    help="The first sheet should contain columns Rotation and Moment"
)

if not uploaded:
    st.info("Upload an Excel file to begin")
    st.stop()

# Read Excel
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

# Compute stiffness
x_tan = rotation[i_tangent]
y_tan = moment[i_tangent]
if x_tan == 0:
    st.error("Rotation at I_TANGENT is zero so Sj,ini is undefined")
    st.stop()

Sj_ini = y_tan / x_tan
Sj = Sj_ini / 2.0

# Mrd intersection and limits for stiffness lines
x_mrd = rot_at_mrd(rotation, moment, mrd)
x_max = np.nanmax(rotation)
x_lim_ini = mrd / Sj_ini if Sj_ini else None
x_lim_sj = mrd / Sj if Sj else None
x_limit = safe_min(x_lim_ini, x_lim_sj, x_mrd if x_mrd is not None else x_max, x_max)

x_line = np.array([0.0, x_limit]) if x_limit is not None else np.array([0.0, x_max])
y_line_ini = Sj_ini * x_line
y_line_sj = Sj * x_line

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(rotation, moment, label="Moment–Rotation", color="black", linewidth=1.8)
ax.plot(x_line, y_line_ini, "--", label=f"Sj,ini ≈ {Sj_ini:.1f} kNm/rad", color="blue")
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

# Results panel
st.subheader("Results")
c3, c4, c5 = st.columns(3)
c3.metric("Sj,ini [kNm/rad]", f"{Sj_ini:.1f}")
c4.metric("Sj [kNm/rad]", f"{Sj:.1f}")
if x_mrd is not None and np.isfinite(x_mrd):
    c5.metric("Rotation at Mrd [rad]", f"{x_mrd:.6f}")
else:
    c5.metric("Rotation at Mrd [rad]", "no intersection")

# Downloads
st.subheader("Download")
# Figure as PNG
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
st.download_button(
    "Download plot (PNG)",
    data=buf.getvalue(),
    file_name="rotational_stiffness.png",
    mime="image/png",
)

# Export computed lines
out_df = pd.DataFrame({
    "Rotation": rotation,
    "Moment": moment
})
st.download_button(
    "Download original data (CSV)",
    data=out_df.to_csv(index=False).encode("utf-8"),
    file_name="input_data.csv",
    mime="text/csv",
)

# Export stiffness line samples
line_df = pd.DataFrame({
    "x_line": x_line,
    "y_line_Sj_ini": y_line_ini,
    "y_line_Sj": y_line_sj
})
st.download_button(
    "Download stiffness lines (CSV)",
    data=line_df.to_csv(index=False).encode("utf-8"),
    file_name="stiffness_lines.csv",
    mime="text/csv",
)


