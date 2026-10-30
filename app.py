"""
BikeWorth - Used Bike Price Prediction Analysis Using Machine Learning
Community Engagement Project, B.Sc. (Data Science), 2026-27.

Run with:   streamlit run app.py

Pages (Table 4.1 of the report):
  Home | Dataset Explorer | Insights | Price Predictor | Model Comparison | Download and Help
"""
from __future__ import annotations

import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import bikeworth_core as core
from bikeworth_core import inr

ACCENT = "#ff5a36"
PAGES = ["Home", "Dataset Explorer", "Insights", "Price Predictor",
         "Model Comparison", "Download and Help"]
WIDE_PAGES = {"Dataset Explorer", "Insights"}

st.set_page_config(page_title="BikeWorth", page_icon="🏍️", layout="wide",
                   initial_sidebar_state="auto")


# --------------------------------------------------------------------------
# Data and model loading (cached)
# --------------------------------------------------------------------------
def _ensure_assets() -> None:
    """Build the dataset and model on first run if the files are not there."""
    if not core.DATA_RAW.exists():
        import gen_data
        gen_data.main()
    if not (core.DATA_CLEAN.exists() and core.MODEL_PATH.exists() and core.METRICS_PATH.exists()):
        raw = pd.read_csv(core.DATA_RAW)
        clean, info = core.prepare_data(raw)
        clean.to_csv(core.DATA_CLEAN, index=False)
        artifact, metrics = core.train_all(clean, info)
        core.save_artifacts(artifact, metrics)


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    _ensure_assets()
    return pd.read_csv(core.DATA_CLEAN)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    _ensure_assets()
    try:
        return core.load_artifacts()
    except Exception:  # e.g. scikit-learn version differs from the one that saved the file
        clean = pd.read_csv(core.DATA_CLEAN)
        raw = pd.read_csv(core.DATA_RAW)
        _, info = core.prepare_data(raw)
        artifact, metrics = core.train_all(clean, info)
        core.save_artifacts(artifact, metrics)
        return artifact, metrics


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
BASE_CSS = f"""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stApp {{ background: #0f1013; }}
.block-container {{ padding-top: 3rem; padding-bottom: 2rem; }}
.bw-top {{ display:flex; justify-content:space-between; align-items:center;
           padding: .4rem 0 .6rem 0; border-bottom: 1px solid #23262d; margin-bottom: .6rem; }}
.bw-brand {{ font-size: 1.25rem; font-weight: 800; color:#fff; letter-spacing:.2px; }}
.bw-brand span {{ color: {ACCENT}; }}
.bw-menu {{ color:#8a8f98; font-size:.85rem; }}
.bw-eyebrow {{ color:{ACCENT}; font-size:.72rem; font-weight:800; letter-spacing:.08em; margin-top:1.2rem; }}
.bw-h1 {{ font-size: 2.5rem; line-height:1.08; font-weight: 800; color:#f2f3f5; margin:.4rem 0 .8rem 0; }}
.bw-lead {{ color:#9aa0aa; font-size:.95rem; line-height:1.45; margin-bottom:1rem; }}
.bw-h2 {{ font-size:1.35rem; font-weight:800; color:#f2f3f5; margin:1.4rem 0 .1rem 0; }}
.bw-sub {{ color:#7d828c; font-size:.78rem; margin-bottom:.6rem; }}
.bw-stat {{ background:#16181d; border:1px solid #23262d; border-radius:12px; padding:.9rem 1rem; margin-bottom:.6rem; }}
.bw-stat b {{ display:block; font-size:2rem; color:#fff; line-height:1.1; }}
.bw-stat small {{ color:#7d828c; font-size:.75rem; }}
.bw-bar-row {{ margin:.55rem 0; }}
.bw-bar-label {{ display:flex; justify-content:space-between; font-size:.85rem; color:#d6d8dd; margin-bottom:.25rem; }}
.bw-bar-label em {{ font-style:normal; color:#8a8f98; font-size:.8rem; }}
.bw-bar {{ height:7px; background:#272a31; border-radius:4px; overflow:hidden; }}
.bw-bar > i {{ display:block; height:100%; background:#5b6272; border-radius:4px; }}
.bw-bar.hl > i {{ background:{ACCENT}; }}
.bw-est-label {{ color:#8a8f98; font-size:.95rem; margin-top:.8rem; }}
.bw-price {{ color:{ACCENT}; font-size:3rem; font-weight:800; line-height:1.1; margin:.1rem 0; }}
.bw-range {{ color:#eceef2; font-size:1.05rem; }}
.bw-meta {{ color:#6f7480; font-size:.78rem; margin-top:.25rem; }}
.bw-note {{ background:#16181d; border:1px solid #23262d; border-radius:10px; color:#9aa0aa;
            font-size:.8rem; padding:.7rem .9rem; margin:.8rem 0; }}
.bw-warn {{ background:#2a1d16; border:1px solid #6b3a22; border-radius:10px; color:#f2b48f;
            font-size:.82rem; padding:.7rem .9rem; margin:.5rem 0; }}
.bw-sim {{ display:flex; justify-content:space-between; background:#16181d; border:1px solid #23262d;
           border-radius:10px; padding:.6rem .9rem; margin:.4rem 0; font-size:.85rem; color:#d6d8dd; }}
.bw-sim b {{ color:{ACCENT}; }}
.bw-small {{ color:#6f7480; font-size:.75rem; }}
.bw-footer {{ border-top:1px solid #23262d; color:#6f7480; font-size:.75rem; margin-top:2rem; padding-top:.8rem; }}
div[data-testid="stRadio"] > div[role="radiogroup"] {{ gap:.25rem .9rem; flex-wrap:wrap; }}
@media (max-width: 640px) {{ .bw-h1 {{ font-size:2rem; }} .bw-price {{ font-size:2.4rem; }} }}
</style>
"""


def inject_css(page: str) -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    width = "1200px" if page in WIDE_PAGES else "760px"
    extra = f".block-container {{ max-width: {width}; }}"
    if page != "Dataset Explorer":
        extra += ('[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
                  '[data-testid="collapsedControl"] { display:none !important; }')
    st.markdown(f"<style>{extra}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def bar_row(label: str, value: float, vmax: float, text: str, highlight: bool = True) -> str:
    width = max(2.0, min(100.0, value / vmax * 100.0)) if vmax else 0
    cls = "bw-bar hl" if highlight else "bw-bar"
    return (f'<div class="bw-bar-row"><div class="bw-bar-label"><span>{label}</span>'
            f'<em>{text}</em></div><div class="{cls}"><i style="width:{width:.1f}%"></i></div></div>')


def show_chart(chart: alt.Chart) -> None:
    """Altair chart that works across Streamlit versions."""
    try:
        st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart, width="stretch")


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        st.dataframe(df, width="stretch", **kwargs)


def go(page: str) -> None:
    st.session_state["page"] = page


def footer() -> None:
    html('<div class="bw-footer">© 2026 BikeWorth · Used bike price prediction analysis</div>')


def short_name(row_name: str, brand: str) -> str:
    return row_name[len(brand):].strip() if row_name.startswith(brand) else row_name


# --------------------------------------------------------------------------
# Page: Home (M1)
# --------------------------------------------------------------------------
def page_home(df, artifact, metrics):
    rf = metrics["models"]["Random Forest"]
    html('<div class="bw-eyebrow">USED BIKE PRICE PREDICTION</div>'
         '<div class="bw-h1">Know what your used bike is really worth.</div>'
         f'<div class="bw-lead">A machine-learning study of {len(df):,} resale listings from '
         f'{df["brand"].nunique()} brands. Enter a few details and get an estimated price with '
         'a realistic range.</div>')
    c1, c2, _ = st.columns([1.1, 1.1, 2])
    c1.button("Predict price →", type="primary", on_click=go, args=("Price Predictor",),
              key="home_predict")
    c2.button("Explore data", on_click=go, args=("Dataset Explorer",), key="home_explore")

    a, b = st.columns(2)
    a.markdown(f'<div class="bw-stat"><b>{len(df):,}</b><small>Listings analysed</small></div>',
               unsafe_allow_html=True)
    b.markdown(f'<div class="bw-stat"><b>{df["brand"].nunique()}</b><small>Brands covered</small></div>',
               unsafe_allow_html=True)
    a.markdown(f'<div class="bw-stat"><b>{rf["R2"]:.2f}</b><small>R² (Random Forest)</small></div>',
               unsafe_allow_html=True)
    b.markdown(f'<div class="bw-stat"><b>{rf["MAPE"]:.1f}%</b><small>Mean abs. % error</small></div>',
               unsafe_allow_html=True)

    html('<div class="bw-h2">What drives the resale price?</div>'
         '<div class="bw-sub">Feature importance, Random Forest</div>')
    top = list(metrics["importance"].items())[:5]
    vmax = top[0][1] / 0.95
    html("".join(bar_row(k, v, vmax, f"{v:.2f}") for k, v in top))
    footer()


# --------------------------------------------------------------------------
# Page: Dataset Explorer (M2)
# --------------------------------------------------------------------------
BAND_OPTIONS = ["All prices", "Under ₹30,000", "₹30,000 to ₹80,000", "₹80,000 and above"]
SORT_OPTIONS = {
    "Selling price: high to low": ("selling_price", False),
    "Selling price: low to high": ("selling_price", True),
    "Age: newest first": ("age_years", True),
    "Age: oldest first": ("age_years", False),
    "Kilometres: low to high": ("km_driven", True),
    "Kilometres: high to low": ("km_driven", False),
}
FILTER_DEFAULTS = {"f_brand": [], "f_city": [], "f_year": (2004, 2025), "f_band": BAND_OPTIONS[0],
                   "f_sort": "Selling price: high to low", "f_search": ""}


def reset_filters():
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v


def page_explorer(df, artifact, metrics):
    with st.sidebar:
        st.markdown("### Filters")
        st.multiselect("Brand", sorted(df["brand"].unique()), key="f_brand")
        st.multiselect("City", sorted(df["city"].unique()), key="f_city")
        st.slider("Model year", 2004, 2025, value=(2004, 2025), key="f_year")
        st.selectbox("Price band", BAND_OPTIONS, key="f_band")
        st.selectbox("Sort by", list(SORT_OPTIONS), key="f_sort")
        st.text_input("Search bike name", key="f_search")
        st.button("Reset", on_click=reset_filters, key="f_reset")

    out = df
    if st.session_state["f_brand"]:
        out = out[out["brand"].isin(st.session_state["f_brand"])]
    if st.session_state["f_city"]:
        out = out[out["city"].isin(st.session_state["f_city"])]
    y0, y1 = st.session_state["f_year"]
    out = out[out["model_year"].between(y0, y1)]
    band = st.session_state["f_band"]
    if band == BAND_OPTIONS[1]:
        out = out[out["selling_price"] < 30000]
    elif band == BAND_OPTIONS[2]:
        out = out[(out["selling_price"] >= 30000) & (out["selling_price"] < 80000)]
    elif band == BAND_OPTIONS[3]:
        out = out[out["selling_price"] >= 80000]
    q = st.session_state["f_search"].strip()
    if q:
        out = out[out["bike_name"].str.contains(q, case=False, regex=False)]
    col, asc = SORT_OPTIONS[st.session_state["f_sort"]]
    out = out.sort_values(col, ascending=asc, kind="stable")
    st.session_state["filtered_df"] = out

    html('<div class="bw-h2" style="margin-top:.2rem">Dataset Explorer</div>')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Listings shown", f"{len(out):,} of {len(df):,}")
    m2.metric("Columns", f"{out.shape[1]}")
    m3.metric("Median price", f"₹ {inr(out['selling_price'].median())}" if len(out) else "–")
    m4.metric("Median age", f"{out['age_years'].median():.0f} years" if len(out) else "–")
    if out.empty:
        st.info("No listings match these filters. Press Reset to start again.")
    else:
        show_df(out.reset_index(drop=True), height=520, column_config={
            "selling_price": st.column_config.NumberColumn("selling_price (₹)", format="%d"),
            "ex_showroom_price": st.column_config.NumberColumn("ex_showroom_price (₹)", format="%d"),
            "km_driven": st.column_config.NumberColumn("km_driven", format="%d"),
            "model_year": st.column_config.NumberColumn("model_year", format="%d"),
        })
    footer()


# --------------------------------------------------------------------------
# Page: Insights (M3)
# --------------------------------------------------------------------------
def chart_distribution(df):
    d = pd.DataFrame({"price_k": df["selling_price"] / 1000.0})
    med = float(d["price_k"].median())
    bars = (alt.Chart(d).mark_bar(color=ACCENT, opacity=0.9)
            .encode(x=alt.X("price_k:Q", bin=alt.Bin(maxbins=40), title="Selling price (₹ thousand)"),
                    y=alt.Y("count():Q", title="Number of listings"))
            .properties(title="Distribution of selling price", height=300))
    rule = (alt.Chart(pd.DataFrame({"m": [med]})).mark_rule(color="#ffffff", strokeDash=[5, 4])
            .encode(x="m:Q"))
    return bars + rule


def chart_price_age(df, r):
    s = df.sample(400, random_state=1).assign(lakh=lambda x: x["selling_price"] / 100000.0)
    return (alt.Chart(s).mark_circle(color=ACCENT, size=36, opacity=0.8)
            .encode(x=alt.X("age_years:Q", title="Age of bike (years)"),
                    y=alt.Y("lakh:Q", scale=alt.Scale(type="log"),
                            title="Selling price (INR lakh, log scale)"),
                    tooltip=["bike_name", "age_years", "selling_price"])
            .properties(title=f"Resale price falls steadily with age (r = {r:.2f})", height=320))


def chart_median_brand(df):
    m = (df.groupby("brand")["selling_price"].median().reset_index()
         .rename(columns={"selling_price": "median_price"}).sort_values("median_price", ascending=False))
    return (alt.Chart(m).mark_bar(color=ACCENT)
            .encode(y=alt.Y("brand:N", sort="-x", title=None),
                    x=alt.X("median_price:Q", title="Median selling price (₹)"),
                    tooltip=["brand", "median_price"])
            .properties(title="Median selling price by brand", height=340))


def chart_listings_brand(df):
    m = df["brand"].value_counts().reset_index()
    m.columns = ["brand", "listings"]
    return (alt.Chart(m).mark_bar(color="#2f5b9c")
            .encode(y=alt.Y("brand:N", sort="-x", title=None),
                    x=alt.X("listings:Q", title="Number of used-bike listings"),
                    tooltip=["brand", "listings"])
            .properties(title=f"Listings per brand ({df['brand'].nunique()} brands, {len(df):,} listings)",
                        height=340))


def page_insights(df, artifact, metrics):
    html('<div class="bw-h2" style="margin-top:.2rem">Insights</div>'
         '<div class="bw-sub">What explains the resale price</div>')
    corr = metrics["correlations"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Age vs price", f"{corr['age_vs_price']:.2f}")
    c2.metric("Ex-showroom vs price", f"{corr['ex_vs_price']:.2f}",
              help=f"Computed on the {corr['n_with_ex']:,} listings that have an ex-showroom price.")
    c3.metric("Max power vs price", f"{corr['power_vs_price']:.2f}")

    choice = st.radio("Chart selector",
                      ["All charts", "Price distribution", "Price vs age",
                       "Median price by brand", "Listings per brand"],
                      horizontal=True, key="insight_choice")
    if choice in ("All charts", "Price distribution"):
        show_chart(chart_distribution(df))
        s = df["selling_price"]
        st.caption(f"Median ₹{inr(s.median())} (dashed line) is below the mean ₹{inr(s.mean())}: "
                   "a few premium bikes pull the mean upwards.")
    if choice in ("All charts", "Price vs age"):
        show_chart(chart_price_age(df, corr["age_vs_price"]))
        st.caption("Sample of 400 listings on a logarithmic price scale.")
    if choice in ("All charts", "Median price by brand"):
        show_chart(chart_median_brand(df))
    if choice == "Listings per brand":
        show_chart(chart_listings_brand(df))

    with st.expander("Price bands by brand (Table 6.3)"):
        bands = pd.cut(df["selling_price"], [0, 29999.99, 79999.99, 1e12],
                       labels=["Under ₹30,000", "₹30,000 to 80,000", "₹80,000 and above"])
        t = pd.crosstab(df["brand"], bands)
        t["Total"] = t.sum(axis=1)
        t = t.sort_values("Total", ascending=False)
        t.loc["Total"] = t.sum()
        show_df(t)
    with st.expander("Ten most expensive listings (Table 6.4)"):
        top = df.sort_values("selling_price", ascending=False).head(10)
        show_df(top[["bike_name", "brand", "selling_price", "max_power_bhp"]]
                .rename(columns={"selling_price": "selling_price (₹)"}).reset_index(drop=True))
    footer()


# --------------------------------------------------------------------------
# Page: Price Predictor (M4)
# --------------------------------------------------------------------------
EXAMPLE = {"p_brand": "Royal Enfield", "p_year": 2018, "p_km": 32000, "p_owner": "1st owner",
           "p_seller": "individual", "p_fuel": "petrol", "p_cc": 346, "p_bhp": 19.8,
           "p_ex": 140000, "p_cond": 4}
PRED_KEYS = list(EXAMPLE)


def use_example():
    for k, v in EXAMPLE.items():
        st.session_state[k] = v
    st.session_state.pop("last_prediction", None)


def clear_form():
    for k in PRED_KEYS:
        st.session_state[k] = None
    st.session_state.pop("last_prediction", None)


def page_predictor(df, artifact, metrics):
    if not any(k in st.session_state for k in PRED_KEYS):  # form opens filled with the example (Fig 7.2)
        for k, v in EXAMPLE.items():
            st.session_state[k] = v
    cats = artifact["prep"]["categories"]
    R = core.INPUT_RANGES

    html('<div class="bw-h1" style="font-size:2rem;margin-top:.4rem">Price Predictor</div>'
         '<div class="bw-sub" style="font-size:.9rem">Enter the bike details to get an estimate</div>')
    c1, c2 = st.columns(2)
    c1.selectbox("Brand", cats["brand"], index=None, key="p_brand", placeholder="Select a brand")
    c2.number_input("Model year", min_value=R["model_year"][0], max_value=R["model_year"][1],
                    value=None, step=1, key="p_year", format="%d")
    c1.number_input("Km driven", min_value=R["km_driven"][0], max_value=R["km_driven"][1],
                    value=None, step=500, key="p_km", format="%d")
    c2.selectbox("Owner type", cats["owner_type"], index=None, key="p_owner", placeholder="Select")
    c1.selectbox("Seller type", cats["seller_type"], index=None, key="p_seller",
                 format_func=str.capitalize, placeholder="Select")
    c2.selectbox("Fuel type", cats["fuel_type"], index=None, key="p_fuel",
                 format_func=str.capitalize, placeholder="Select")
    c1.number_input("Engine (cc)", min_value=R["engine_cc"][0], max_value=R["engine_cc"][1],
                    value=None, step=1, key="p_cc", format="%d")
    c2.number_input("Max power (bhp)", min_value=float(R["max_power_bhp"][0]),
                    max_value=float(R["max_power_bhp"][1]), value=None, step=0.1,
                    key="p_bhp", format="%.1f")
    c1.number_input("Ex-showroom price (₹)", min_value=R["ex_showroom_price"][0],
                    max_value=R["ex_showroom_price"][1], value=None, step=1000,
                    key="p_ex", format="%d")
    c2.number_input("Condition rating (1-5)", min_value=R["condition_rating"][0],
                    max_value=R["condition_rating"][1], value=None, step=1,
                    key="p_cond", format="%d")

    b1, b2, b3, _ = st.columns([1.2, 1.1, 0.8, 1.6])
    predict = b1.button("Predict price", type="primary", key="p_go")
    b2.button("Use example", on_click=use_example, key="p_example")
    b3.button("Clear", on_click=clear_form, key="p_clear")
    html('<div class="bw-small" style="margin-top:.5rem">Estimate only. Not an official valuation.</div>')

    if predict:
        s = st.session_state
        inp = {"brand": s["p_brand"], "model_year": s["p_year"], "km_driven": s["p_km"],
               "owner_type": s["p_owner"], "seller_type": s["p_seller"], "fuel_type": s["p_fuel"],
               "engine_cc": s["p_cc"], "max_power_bhp": s["p_bhp"],
               "ex_showroom_price": s["p_ex"], "condition_rating": s["p_cond"]}
        missing = [k for k, v in inp.items() if v is None]
        if missing:
            st.session_state.pop("last_prediction", None)
            st.error("Please fill in all ten fields before predicting. Missing: "
                     + ", ".join(m.replace("_", " ") for m in missing) + ".")
        else:
            problems = core.validate_inputs(inp)
            if problems:
                st.session_state.pop("last_prediction", None)
                st.error("Check your input:\n\n" + "\n".join(f"- {p}" for p in problems))
            else:
                res = core.predict_price(artifact, inp)
                st.session_state["last_prediction"] = {"inputs": inp, "result": res}

    lp = st.session_state.get("last_prediction")
    if lp:
        show_result(df, artifact, metrics, lp["inputs"], lp["result"])
    footer()


def show_result(df, artifact, metrics, inp, res):
    est = res["estimate"]
    html('<div class="bw-est-label">Estimated resale price</div>'
         f'<div class="bw-price">₹ {inr(est)}</div>'
         f'<div class="bw-range">Likely range: ₹ {inr(res["low"])} to ₹ {inr(res["high"])}</div>'
         f'<div class="bw-meta">{inp["brand"]} · {inp["model_year"]} · {inr(inp["km_driven"])} km · '
         f'{inp["owner_type"]}</div>')
    html('<div class="bw-note">Estimate, not an official valuation. Rare brands have fewer '
         'listings, so the range is wider.</div>')

    seg = metrics["segments_mape"]
    band = core.band_of(est)
    html(f'<div class="bw-small">Typical error for bikes in the {band} band: about '
         f'{seg[band]:.0f}%.</div>')
    if res["brand_listings"] < 100:
        html(f'<div class="bw-warn">Only {res["brand_listings"]} {inp["brand"]} listings are in the '
             f'data, so this estimate rests on few similar bikes. Error for brands with fewer than '
             f'100 listings is about {seg["Brands with fewer than 100 listings"]:.0f}%; treat the '
             'range with extra caution.</div>')
    if inp["ex_showroom_price"] is not None and df["ex_showroom_price"].isna().mean() > 0.2:
        st.caption("About a quarter of the training listings had no ex-showroom price, which was "
                   "imputed; a well-known original price makes the estimate more reliable.")

    html('<div class="bw-h2">Model comparison (test R²)</div>')
    models = metrics["models"]
    html("".join(bar_row(n, m["R2"], 1.0, f"{m['R2']:.2f}", highlight=(n == "Random Forest"))
                 for n, m in models.items()))

    html('<div class="bw-h2">Similar listings used for context</div>')
    sim = core.similar_listings(df, inp, k=3)
    rows = []
    for _, r in sim.iterrows():
        rows.append(f'<div class="bw-sim"><span>{short_name(r["bike_name"], r["brand"])} · '
                    f'{int(r["model_year"])} · {inr(r["km_driven"])} km</span>'
                    f'<b>₹ {inr(r["selling_price"])}</b></div>')
    html("".join(rows))

    # keep for the Download page
    row = {**inp, "estimated_price": est, "range_low": res["low"], "range_high": res["high"]}
    st.session_state["last_prediction_row"] = row


# --------------------------------------------------------------------------
# Page: Model Comparison (M5)
# --------------------------------------------------------------------------
def page_models(df, artifact, metrics):
    html('<div class="bw-h2" style="margin-top:.2rem">Model Comparison</div>'
         f'<div class="bw-sub">Six regression models trained on {metrics["n_train"]:,} listings and '
         f'tested on {metrics["n_test"]:,}</div>')
    metric = st.radio("Metric", ["R-squared", "MAE", "RMSE"], horizontal=True, key="model_metric")
    key = {"R-squared": "R2", "MAE": "MAE", "RMSE": "RMSE"}[metric]
    models = metrics["models"]
    best = (max if key == "R2" else min)(models, key=lambda n: models[n][key])

    tbl = pd.DataFrame({
        "Model": list(models),
        "MAE (₹)": [round(m["MAE"]) for m in models.values()],
        "RMSE (₹)": [round(m["RMSE"]) for m in models.values()],
        "R² (test)": [round(m["R2"], 2) for m in models.values()],
        "R² (5-fold CV)": [round(m["R2_CV"], 2) for m in models.values()],
        "MAPE (%)": [round(m["MAPE"], 1) for m in models.values()],
    })
    tbl["Model"] = [n + (" ✓ chosen" if n == "Random Forest" else "") for n in tbl["Model"]]
    show_df(tbl, hide_index=True)

    chart_df = pd.DataFrame({"Model": list(models), "value": [m[key] for m in models.values()],
                             "best": [n == best for n in models]})
    title = {"R2": "R-squared (test set, higher is better)", "MAE": "Mean absolute error, ₹ (lower is better)",
             "RMSE": "Root mean squared error, ₹ (lower is better)"}[key]
    chart = (alt.Chart(chart_df).mark_bar()
             .encode(y=alt.Y("Model:N", sort=None, title=None),
                     x=alt.X("value:Q", title=metric),
                     color=alt.condition("datum.best", alt.value(ACCENT), alt.value("#5b6272")),
                     tooltip=["Model", "value"])
             .properties(title=title, height=260))
    show_chart(chart)

    rf = models["Random Forest"]
    st.markdown(
        f"The Random Forest is within about ₹{inr(round(rf['MAE'], -2))} of the listed price on a typical "
        f"test listing (mean absolute percentage error {rf['MAPE']:.1f}%). Gradient boosting is close "
        "behind; the forest was chosen because its accuracy was higher, it needs less tuning, and the "
        "spread of its 200 trees gives a natural likely range.")

    html('<div class="bw-h2">Where the estimate is weaker</div>')
    seg = pd.DataFrame({"Segment": list(metrics["segments_mape"]),
                        "Mean abs. % error": [round(v, 1) for v in metrics["segments_mape"].values()]})
    show_df(seg, hide_index=True)
    st.caption(f"Random Forest without the ex-showroom price field reaches R² "
               f"{metrics['rf_r2_without_ex_showroom']:.2f}, compared with {rf['R2']:.2f} with it.")

    html('<div class="bw-h2">Feature importance</div>')
    imp = list(metrics["importance"].items())
    vmax = imp[0][1] / 0.95
    html("".join(bar_row(k, v, vmax, f"{v:.2f}") for k, v in imp[:8]))
    st.caption(f"Age, ex-showroom price and maximum power carry {metrics['top3_importance_share']:.0%} "
               "of the importance.")

    with st.expander("Model settings and standardisation parameters (Table 8.2)"):
        p = metrics["rf_params"]
        st.write(f"Random Forest: {p['n_estimators']} trees, maximum depth {p['max_depth']}, at least "
                 f"{p['min_samples_leaf']} listings per leaf. The likely range is the 10th to 90th "
                 "percentile of the 200 tree outputs.")
        sc = pd.DataFrame([{"Input field": k, "Mean (training rows)": round(v[0], 1),
                            "Standard deviation": round(v[1], 1)} for k, v in metrics["scaler"].items()])
        show_df(sc, hide_index=True)
    footer()


# --------------------------------------------------------------------------
# Page: Download and Help (M6)
# --------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first = df.iloc[0]
    for i, c in enumerate(df.columns, 1):
        kind = "number" if pd.api.types.is_numeric_dtype(df[c]) else "text"
        rows.append({"No.": i, "Field": c + (" (derived)" if c == "age_years" else ""), "Type": kind,
                     "Missing": int(df[c].isna().sum()), "Example (first record)": str(first[c]),
                     "Distinct": int(df[c].nunique())})
    return pd.DataFrame(rows)


def integrity_checks(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    n_dup = metrics.get("data_info", {}).get("duplicates_removed", 0)
    over = int((df["selling_price"] > df["ex_showroom_price"]).sum())
    miss_ex = int(df["ex_showroom_price"].isna().sum())
    miss_km = int(df["mileage_kmpl"].isna().sum())
    checks = [
        ("Record count", "5,000 listings expected", f"{len(df):,} records", len(df) == 5000),
        ("Field count", "15 fields per record", f"{df.shape[1]} fields", df.shape[1] == 15),
        ("Price validity", "Selling price greater than 0", f"{int((df['selling_price'] <= 0).sum())} violations",
         (df["selling_price"] > 0).all()),
        ("Price type", "Price stored as a number", "All numeric" if pd.api.types.is_numeric_dtype(df["selling_price"]) else "Not numeric",
         pd.api.types.is_numeric_dtype(df["selling_price"])),
        ("Brand completeness", "Brand not empty", f"{int(df['brand'].isna().sum())} missing", df["brand"].notna().all()),
    ]
    out = [{"No.": i, "Check": a, "Rule": b, "Result": c, "Status": "Pass" if ok else "Fail"}
           for i, (a, b, c, ok) in enumerate(checks, 1)]
    out.append({"No.": 6, "Check": "Duplicate records", "Rule": "No two identical listings",
                "Result": f"{n_dup} exact duplicates found and removed", "Status": "Warning" if n_dup else "Pass"})
    out.append({"No.": 7, "Check": "Ex-showroom price", "Rule": "Nullable, reported where known",
                "Result": f"{miss_ex:,} missing ({miss_ex / len(df):.0%})", "Status": "Warning" if miss_ex else "Pass"})
    out.append({"No.": 8, "Check": "Mileage", "Rule": "Nullable",
                "Result": f"{miss_km} missing ({miss_km / len(df):.0%})", "Status": "Warning" if miss_km else "Pass"})
    out.append({"No.": 9, "Check": "Power range", "Rule": "Greater than 0 bhp",
                "Result": f"{df['max_power_bhp'].min():.1f} to {df['max_power_bhp'].max():.1f} bhp",
                "Status": "Pass" if df["max_power_bhp"].min() > 0 else "Fail"})
    out.append({"No.": 10, "Check": "Age range", "Rule": "0 to 30 years",
                "Result": f"{df['age_years'].min()} to {df['age_years'].max()} years",
                "Status": "Pass" if df["age_years"].between(0, 30).all() else "Fail"})
    out.append({"No.": 11, "Check": "Kilometres range", "Rule": "0 to 3,00,000 km",
                "Result": f"{inr(df['km_driven'].min())} to {inr(df['km_driven'].max())} km",
                "Status": "Pass" if df["km_driven"].between(0, 300000).all() else "Fail"})
    out.append({"No.": 12, "Check": "Resale price", "Rule": "Selling price not above ex-showroom price",
                "Result": f"{over} listings above it", "Status": "Fail" if over else "Pass"})
    return pd.DataFrame(out)


def page_help(df, artifact, metrics):
    html('<div class="bw-h2" style="margin-top:.2rem">Download and Help</div>')
    t1, t2, t3, t4, t5 = st.tabs(["Download", "How it works", "Data dictionary",
                                  "Data quality", "Limits"])
    with t1:
        filt = st.session_state.get("filtered_df", df)
        st.write(f"**Filtered table** ({len(filt):,} listings). Set filters on the Dataset Explorer page.")
        st.download_button("Download filtered table (CSV)", to_csv_bytes(filt),
                           file_name="bikeworth_filtered_listings.csv", mime="text/csv", key="dl_filtered")
        st.write("**Latest prediction**")
        row = st.session_state.get("last_prediction_row")
        if row:
            show_df(pd.DataFrame([row]), hide_index=True)
            st.download_button("Download latest prediction (CSV)", to_csv_bytes(pd.DataFrame([row])),
                               file_name="bikeworth_prediction.csv", mime="text/csv", key="dl_pred")
        else:
            st.info("No prediction yet. Make one on the Price Predictor page.")
    with t2:
        rf = metrics["models"]["Random Forest"]
        st.markdown(f"""
**Data.** {len(df):,} used-bike listings from {df['brand'].nunique()} brands and
{df['city'].nunique()} cities, with 15 fields each. Exact duplicates were removed and the age of each
bike was derived from its model year.

**Features.** Ten inputs: brand, model year (turned into age), kilometres driven, owner type, seller
type, fuel type, engine capacity, maximum power, ex-showroom price and condition rating. Categories are
one-hot encoded and the six numeric fields are standardised with the training means and standard
deviations. A missing ex-showroom price is replaced by the brand's median.

**Model.** Six regressors were compared (linear, ridge, KNN with K = 7, decision tree, random forest,
gradient boosting) on a 4,000 / 1,000 train / test split with 5-fold cross-validation. The Random Forest
(200 trees, depth 18, at least two listings per leaf) was best: R² {rf['R2']:.2f}, mean absolute error
₹{inr(rf['MAE'])}, mean absolute percentage error {rf['MAPE']:.1f}%.

**Estimate and range.** The estimate is the mean of the 200 trees. The likely range is the 10th to 90th
percentile of the tree outputs, so a wide range means the trees disagree. It is a measure of
disagreement, not a statistical guarantee.
""")
    with t3:
        show_df(data_dictionary(df), hide_index=True)
    with t4:
        show_df(integrity_checks(df, metrics), hide_index=True)
        st.caption("Warnings and the failure do not stop the app, but they affect how far the estimate "
                   "can be trusted: a quarter of the listings have no ex-showroom price.")
    with t5:
        st.markdown("""
- The figure is an **estimate, not an official valuation**, and not a guaranteed selling price.
- The data are asking prices from one snapshot, not completed sales, and are not updated automatically.
- The model cannot see accident history, service records or the physical state of a particular bike.
- Budget bikes (under ₹30,000) and brands with fewer than 100 listings (Mahindra, Jawa, Triumph) have
  fewer similar listings, so their estimates are less reliable. A warning is shown for rare brands.
- The likely range has not been checked against actual sale prices.
- Inputs are validated against the ranges seen in the dataset (for example model year 2004 to 2025).
""")
    footer()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.session_state.setdefault("page", "Home")
    page = st.session_state["page"]
    inject_css(page)
    html(f'<div class="bw-top"><div class="bw-brand">Bike<span>Worth</span></div>'
         f'<div class="bw-menu">{page}</div></div>')
    st.radio("Menu", PAGES, horizontal=True, key="page", label_visibility="collapsed")

    df = load_data()
    artifact, metrics = load_model()
    {"Home": page_home, "Dataset Explorer": page_explorer, "Insights": page_insights,
     "Price Predictor": page_predictor, "Model Comparison": page_models,
     "Download and Help": page_help}[st.session_state["page"]](df, artifact, metrics)


main()
