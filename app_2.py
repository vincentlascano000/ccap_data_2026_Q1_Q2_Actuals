import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="CCAP — Method C (Dampened)", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")

BANK_COLORS = {
    "UB": "#f28e2b",
    "BDO": "#4169E1",
    "BPI": "#d62728",
    "SECBANK": "#4CAF50",
    "MB": "#3B5B8A",
    "RCBC": "#7ec8e3",
}

# =========================================================
# QUARTER HELPERS
# =========================================================
def parse_quarter_dt(q):
    s = str(q).strip().upper()
    quarter = int(s[0])
    year = 2000 + int(s[2:])
    return pd.Period(year=year, quarter=quarter, freq="Q").to_timestamp(how="end")

def fmt_q(p):
    return f"{p.quarter}Q{str(p.year)[-2:]}"

# =========================================================
# UI — LEVERS
# =========================================================
st.title("CCAP — Method C (Recent‑Weighted + Seasonal Dampening)")

decay = st.sidebar.slider(
    "Recency weighting",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="1.0 = all years equal; lower = trust recent years more"
)

lam = st.sidebar.slider(
    "Seasonality strength (λ)",
    min_value=0.0, max_value=1.0, value=0.6, step=0.05,
    help="1.0 = full seasonal peaks/troughs; lower = smoother, more realistic swings"
)

# =========================================================
# LOAD DATA
# =========================================================
raw = pd.read_csv(RAW_URL).rename(columns={
    "QUARTER": "quarter",
    "BANK": "bank",
    "Purchase Sales (in Bn)": "ps",
    "Cards in Force (in Bn)": "cif",
    "Sales / CIF ('000)": "spc",
})

raw = raw[["quarter", "bank", "ps", "cif", "spc"]]
raw["quarter_dt"] = raw["quarter"].apply(parse_quarter_dt)

for c in ["ps", "cif", "spc"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

panel = (
    raw.dropna()
       .sort_values(["bank", "quarter_dt"])
       .reset_index(drop=True)
)

banks = panel["bank"].unique().tolist()
banks_pick = st.multiselect("Banks", banks, default=banks)
panel = panel[panel["bank"].isin(banks_pick)]

# =========================================================
# PER‑BANK ADJUSTMENT LEVERS (±10 ppt each)
# =========================================================
st.sidebar.header("Per‑Bank Adjustment (ppt)")

bank_adjustments = {}
for b in banks_pick:
    bank_adjustments[b] = st.sidebar.slider(
        f"{b}", min_value=-10.0, max_value=10.0,
        value=0.0, step=0.05,
        help=f"Growth adjustment applied only to {b}"
    ) / 100

# =========================================================
# RECENT‑WEIGHTED + DAMPENED SEASONAL PROFILE
# =========================================================
def weighted_seasonal(gb, decay, lam):
    """
    Recency-weighted seasonal growth per quarter-of-year,
    then dampened toward the cross-quarter average by factor lam.
    lam=1 -> full seasonality; lam=0 -> flat (no seasonality).
    """
    g = gb.assign(
        q=gb["quarter_dt"].dt.quarter,
        yr=gb["quarter_dt"].dt.year,
        d_ps=gb["ps"].pct_change(),
        d_cif=gb["cif"].pct_change(),
        d_spc=gb["spc"].pct_change(),
    ).dropna(subset=["d_ps", "d_cif", "d_spc"])

    if g.empty:
        return pd.DataFrame(columns=["d_ps", "d_cif", "d_spc"])

    max_yr = g["yr"].max()
    g["w"] = np.power(decay, (max_yr - g["yr"]))

    def wavg(sub, col):
        w = sub["w"].to_numpy()
        x = sub[col].to_numpy()
        return np.average(x, weights=w) if w.sum() > 0 else np.nan

    rows = {}
    for q, sub in g.groupby("q"):
        rows[q] = {c: wavg(sub, c) for c in ["d_ps", "d_cif", "d_spc"]}
    seas = pd.DataFrame(rows).T  # index = quarter-of-year

    # ✅ SEASONAL DAMPENING: shrink swings toward the mean
    for col in ["d_ps", "d_cif", "d_spc"]:
        avg = seas[col].mean()
        seas[col] = avg + lam * (seas[col] - avg)

    return seas

# =========================================================
# FIT COEFFICIENTS (uses dampened baseline)
# =========================================================
def fit_uplift(df, decay, lam):
    g = df.copy()
    g["q"] = g["quarter_dt"].dt.to_period("Q").dt.quarter
    g["d_ps"]  = g.groupby("bank")["ps"].pct_change()
    g["d_cif"] = g.groupby("bank")["cif"].pct_change()
    g["d_spc"] = g.groupby("bank")["spc"].pct_change()

    g_base_list = []
    for _, gb in g.groupby("bank"):
        seas = weighted_seasonal(gb, decay, lam)["d_ps"]
        g_base_list.append(gb["q"].map(seas))
    g["g_base"] = pd.concat(g_base_list).sort_index()

    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps", "d_cif", "d_spc"])
    X = np.column_stack([np.ones(len(fit)), fit["d_cif"], fit["d_spc"]])
    y = fit["r_ps"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

alpha_raw, beta_cif, beta_spc = fit_uplift(panel, decay, lam)

# =========================================================
# METHOD C PROJECTION
# =========================================================
def project_method_c(gb, decay, lam):
    last = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last.year) * 4 + (TARGET_END.quarter - last.quarter)

    ps  = gb.iloc[-1]["ps"]
    cif = gb.iloc[-1]["cif"]
    spc = gb.iloc[-1]["spc"]
    bank = gb.iloc[0]["bank"]

    alpha_bank = alpha_raw + bank_adjustments.get(bank, 0.0)
    seasonal = weighted_seasonal(gb, decay, lam)

    rows = []
    for h in range(1, H + 1):
        t = last + h
        q = t.quarter

        g_base = seasonal.loc[q, "d_ps"]  if q in seasonal.index else 0
        d_cif  = seasonal.loc[q, "d_cif"] if q in seasonal.index else 0
        d_spc  = seasonal.loc[q, "d_spc"] if q in seasonal.index else 0

        g_ps = g_base + (alpha_bank + beta_cif * d_cif + beta_spc * d_spc)
        g_ps = np.clip(g_ps, -0.3, 0.3)

        ps  *= (1 + g_ps)
        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        rows.append({
            "quarter_dt": t.to_timestamp(how="end"),
            "quarter_label": fmt_q(t),
            "bank": bank,
            "ps": ps,
            "scenario": "Method C",
        })

    return pd.DataFrame(rows)

proj = pd.concat(
    [
        project_method_c(panel[panel["bank"] == b], decay, lam)
        for b in banks_pick
        if panel[panel["bank"] == b].shape[0] >= 3
    ],
    ignore_index=True,
)

# =========================================================
# DISPLAY
# =========================================================
hist = panel.assign(
    quarter_label=panel["quarter"],
    scenario="Actual"
)[["quarter_label", "bank", "ps", "scenario"]]

plot_df = pd.concat([hist, proj], ignore_index=True)

chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("quarter_label:N", sort=alt.SortField("quarter_dt")),
        y=alt.Y("ps:Q", title="Purchase Sales (Bn)"),
        color=alt.Color(
            "bank:N",
            scale=alt.Scale(
                domain=list(BANK_COLORS.keys()),
                range=list(BANK_COLORS.values()),
            ),
        ),
        strokeDash=alt.condition(
            alt.datum.scenario == "Actual",
            alt.value([0]),
            alt.value([6, 4]),
        ),
    )
)

st.altair_chart(chart, use_container_width=True)

# =========================================================
# STAKEHOLDER PANEL
# =========================================================
adj_rows = "\n".join(
    f"| {b} | `{bank_adjustments[b]*100:+.2f} ppt` |"
    for b in banks_pick
)

st.markdown(f"""
### Growth Formula (Method C — Dampened Seasonality)

$$
\\Delta PS
=
\\big[\\bar{{g}} + \\lambda\\,(G_{{quarter}} - \\bar{{g}})\\big]
+
(\\alpha + \\text{{bank adj}})
+
\\beta_{{CIF}}\\,\\Delta CIF
+
\\beta_{{SPC}}\\,\\Delta(Sales/CIF)
$$

---

### Levers

| Lever | Value | Meaning |
|------|------|---------|
| Recency weighting | `{decay:.2f}` | How much recent years drive seasonality |
| **Seasonality strength (λ)** | `{lam:.2f}` | 1.0 = full peaks/troughs, 0.0 = flat |

### Estimated Parameters

| Component | Value |
|---------|-------|
| Intercept (α, raw) | `{alpha_raw:.4f}` |
| β (Cards in Force) | `{beta_cif:.4f}` |
| β (Sales / CIF) | `{beta_spc:.4f}` |

### Per‑Bank Adjustments

| Bank | Adjustment |
|------|-----------|
{adj_rows}

---

### How Seasonality Dampening Works

• **λ = 1.0** → keeps the full historical seasonal pattern (sharp peaks and troughs)  
• **λ = 0.6** (current) → softens swings by 40%, toward a more realistic shape  
• **λ = 0.0** → removes seasonality entirely (every quarter grows at the average)

The dampening **preserves the average growth trend** — it only compresses the
size of the quarter‑to‑quarter swings, preventing unrealistically high peaks
and low troughs across **all** future quarters and banks.
""")
