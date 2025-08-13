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
1) Try the **very early** part of the curve: points with \(M \le p\\% \cdot M_{max}\) for **p = 0.5–12%**.
2) If that is too sparse (some data jump up fast), fall back to the **first K points by rotation** (K grows from 3 to 10) and pick the **smallest K** that is linear and stable.

**What is p%?**  
If peak moment is 1000 kNm and **p = 2%**, we only use points with **M ≤ 20 kNm** to compute the slope.
        """
    )

# ---------- Helpers ----------
def ols_slope_through_origin(x, y):
    xx = np.asarray(x, float); yy = np.asarray(y, float)
    denom = float(np.dot(xx, xx))
    if denom == 0: return None
    return float(np.dot(xx, yy) / denom)

def r2_linear(y_true, y_fit):
    y = np.asarray(y_true, float); yf = np.asarray(y_fit, float)
    ss_res = float(np.sum((y - yf) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def find_rotation_at_mrd(rot, mom, mrd):
    s = mom - mrd
    hits = np.where(np.isclose(s, 0.0, atol=1e-12))[0]
    if hits.size: return float(rot[hits[-1]])
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0: return None
    i = int(idx[-1]); x0, x1 = rot[i], rot[i+1]; y0, y1 = mom[i], mom[i+1]
    return float(x0 + (mrd - y0) * (x1 - x0) / (y1 - y0))

def window_by_percent(rot, mom, p, theta_min=1e-9):
    """Return (k, r2, mask) using points with M <= p% of Mmax and rotation > theta_min."""
    Mmax = float(np.max(mom))
    lim  = (p / 100.0) * Mmax
    mask = (mom <= lim) & (rot > theta_min)
    xx, yy = rot[mask], mom[mask]
    if xx.size < 2: return None, None, mask
    k = ols_slope_through_origin(xx, yy)
    if k is None or not np.isfinite(k): return None, None, mask
    r2 = r2_linear(yy, k * xx)
    return k, r2, mask

def window_by_prefix(rot, mom, K, theta_min=1e-9):
    """Return (k, r2, mask) on the first K positive-rotation points (smallest rotations)."""
    pos = rot > theta_min
    xx = rot[pos][:K]; yy = mom[pos][:K]
    mask = np.zeros_like(rot, dtype=bool)
    if xx.size >= 2:
        # mark the first K positive-rotation points used
        idx = np.where(pos)[0][:xx.size]
        mask[idx] = True
        k = ols_slope_through_origin(xx, yy)
        if k is None or not np.isfinite(k): return None, None, mask
        r2 = r2_linear(yy, k * xx)
        return k, r2, mask
    return None, None, mask

def choose_auto_window(rot, mom,
                       r2_strict=0.998, r2_loose=0.995,
                       sens_strict=0.03, sens_loose=0.05):
    """
    Hybrid auto:
      A) p-grid (very low): p = 0.5, 0.75, 1.0, 1.5, 2..12 %. Prefer SMALLEST p passing:
           - enough points (min_pts ~ 10% of N, clamp [4,10])
           - R² >= r2_strict (else r2_loose)
           - slope stability vs immediately smaller accepted window: Δk/k <= sens_strict (else sens_loose)
         Fallback A: smallest p with highest R² (>=3 pts)
      B) If A fails (too few points), try rotation-prefix K = 3..10 (prefer SMALLEST K) with same R²/stability rules.
         Fallback B: smallest K with highest R² (>=3 pts)
    Returns {k, r2, mask, reason}.
    """
    N = len(rot)
    min_pts = int(max(4, min(10, round(0.10 * N))))

    # ---- A) low p% sweep (prefer smallest p that passes) ----
    p_grid = [0.5, 0.75, 1.0, 1.5] + list(range(2, 13))
    prev_k = None
    strict_ok = None; loose_ok = None; best_A = None

    for p in p_grid:
        k, r2, mask = window_by_percent(rot, mom, p, theta_min=1e-9)
        n = int(mask.sum()) if mask is not None else 0

        # track best R² among A (fallback A), require >=3 points
        if k is not None and n >= 3:
            if best_A is None or r2 > best_A["r2"] or (r2 == best_A["r2"] and p < best_A["p"]):
                best_A = {"p": p, "k": k, "r2": r2, "mask": mask, "reason": "auto A: best R² fallback"}

        # acceptance checks
        if k is not None and n >= min_pts:
            sens = None if prev_k is None or prev_k == 0 else abs(k - prev_k) / abs(prev_k)
            if sens is not None and r2 is not None:
                if r2 >= r2_strict and sens <= sens_strict and strict_ok is None:
                    strict_ok = {"k": k, "r2": r2, "mask": mask, "reason": f"auto A (strict), p≤{p}%"}
                elif r2 >= r2_loose and sens <= sens_loose and loose_ok is None:
                    loose_ok = {"k": k, "r2": r2, "mask": mask, "reason": f"auto A (loose), p≤{p}%"}

        if k is not None:
            prev_k = k

    if strict_ok is not None: return strict_ok
    if loose_ok  is not None: return loose_ok
    if best_A   is not None and best_A["k"] is not None:
        return {"k": best_A["k"], "r2": best_A["r2"], "mask": best_A["mask"], "reason": f"{best_A['reason']}, p≤{best_A['p']}%"}

    # ---- B) rotation-prefix sweep (prefer smallest K that passes) ----
    prev_k = None
    strict_ok = None; loose_ok = None; best_B = None
    for K in range(3, 11):  # 3..10 points
        k, r2, mask = window_by_prefix(rot, mom, K, theta_min=1e-9)
        n = int(mask.sum()) if mask is not None else 0

        if k is not None and n >= 3:
            if best_B is None or r2 > best_B["r2"] or (r2 == best_B["r2"] and K < best_B["K"]):
                best_B = {"K": K, "k": k, "r2": r2, "mask": mask, "reason": "auto B: best R² fallback"}

        if k is not None and n >= min_pts:
            sens = None if prev_k is None or prev_k == 0 else abs(k - prev_k) / abs(prev_k)
            if sens is not None and r2 is not None:
                if r2 >= r2_strict and sens <= sens_strict and strict_ok is None:
                    strict_ok = {"k": k, "r2": r2, "mask": mask, "reason": f"auto B (strict), first K={K}"}
                elif r2 >= r2_loose and sens <= sens_loose and loose_ok is None:
                    loose_ok = {"k": k, "r2": r2, "mask": mask, "reason": f"auto B (loose), first K={K}"}

        if k is not None:
            prev_k = k

    if strict_ok is not None: return strict_ok
    if loose_ok  is not None: return loose_ok
    if best_B   is not None:
        return {"k": best_B["k"], "r2": best_B["r2"], "mask": best_B["mask"], "reason": f"{best_B['reason']}, K={best_B['K']}"}

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
if mode == "Auto (recommended)":
    choice = choose_auto_window(x, y)
    if choice is None:
        st.error("Could not determine an initial window. Try Manual p% (e.g., 1–3%).")
        st.stop()
    Sj_ini = choice["k"]; used_r2 = choice["r2"]; used_mask = choice["mask"]; note = choice["reason"]
else:
    k, r2, mask_p = window_by_percent(x, y, p_manual, theta_min=1e-9)
    if k is None:
        st.error("Not enough points in the selected window. Increase p% slightly.")
        st.stop()
    Sj_ini = k; used_r2 = r2; used_mask = mask_p; note = f"manual, p≤{p_manual:g}%"

Sj = Sj_ini / 2.0

# ---------- Mrd & lines ----------
def _clip_limit(x_end_ini, x_end_sj, x_mrd, x_max):
    vals = [v for v in [x_end_ini, x_end_sj, (x_mrd if x_mrd is not None else x_max), x_max] if v is not None and np.isfinite(v)]
    return float(np.min(vals)) if vals else x_max

x_mrd = find_rotation_at_mrd(x, y, mrd)
x_max = float(np.max(x)) if x.size else 0.0
x_end_ini = (mrd / Sj_ini) if Sj_ini else x_max
x_end_sj  = (mrd / Sj)     if Sj     else x_max
x_limit = _clip_limit(x_end_ini, x_end_sj, x_mrd, x_max)

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
c3.metric("R² (window)", f"{used_r2:.5f}" if used_r2 is not None else "n/a")
c4.metric("Rotation at Mrd [rad]", f"{find_rotation_at_mrd(x, y, mrd):.6f}" if x.size else "n/a")
