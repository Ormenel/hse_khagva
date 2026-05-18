
import os
import requests
import streamlit as st
import plotly.graph_objects as go


API_URL = os.getenv("API_URL", "http://localhost:8000")

AVAILABLE_MODELS = ["xgb", "lgb", "rf", "lr", "sgd", "calibrated"]

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
]

DEFAULT_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
DEFAULT_RATES = [0.045, 0.046, 0.047, 0.048, 0.048,
                 0.047, 0.046, 0.045, 0.043, 0.042]

def _try_api(payload: dict):

    try:
        r = requests.post(f"{API_URL}/oas/compute", json=payload, timeout=180)
        if r.status_code == 200:
            return r.json()
        st.error(f"Backend error {r.status_code}: {r.text}")
    except requests.ConnectionError:
        st.error(f"Cannot connect to backend at {API_URL}. Is it running?")
    return None


def _fetch_loaded_models():

    try:
        r = requests.get(f"{API_URL}/oas/models", timeout=5)
        if r.status_code == 200:
            return r.json().get("loaded", AVAILABLE_MODELS)
    except requests.RequestException:
        pass
    return AVAILABLE_MODELS


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_latest_par_curve():
    try:
        r = requests.get(f"{API_URL}/oas/par_curve/latest", timeout=15)
        if r.status_code == 200:
            d = r.json()
            return d.get("date"), d["tenors"], d["rates"]
    except requests.RequestException:
        pass
    return None, DEFAULT_TENORS, DEFAULT_RATES


def _fmt_tenor(t: float) -> str:
    return f"{int(t)}" if float(t).is_integer() else f"{t:g}"


def _fmt_rate(r: float) -> str:
    return f"{r:.4f}"


def _plot_rate_paths(times, paths, mean, p05, p95):

    fig = go.Figure()

    # percentile band
    if p05 and p95:
        fig.add_trace(go.Scatter(
            x=list(times) + list(times)[::-1],
            y=[r * 100 for r in p95] + [r * 100 for r in p05][::-1],
            fill="toself",
            fillcolor="rgba(52,152,219,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5–95% band",
            hoverinfo="skip",
        ))

    # Rate paths
    for i, path in enumerate(paths):
        fig.add_trace(go.Scatter(
            x=times, y=[r * 100 for r in path],
            mode="lines",
            line=dict(color="rgba(52,152,219,0.25)", width=1),
            name="Sampled path",
            showlegend=(i == 0),
            hoverinfo="skip",
        ))

    # Mean path on top
    if mean:
        fig.add_trace(go.Scatter(
            x=times, y=[r * 100 for r in mean],
            mode="lines",
            line=dict(color="#e74c3c", width=2.5),
            name="Mean",
        ))

    fig.update_layout(
        title=f"Hull-White Short-Rate Paths",
        xaxis_title="Time (years)",
        yaxis_title="Short Rate (%)",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def _plot_yield_curve(tenors, rates):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenors, y=[r * 100 for r in rates],
        mode="lines+markers", name="Zero Curve",
        line=dict(color="#3498db", width=2),
    ))
    fig.update_layout(
        title="Input Yield Curve",
        xaxis_title="Maturity (years)",
        yaxis_title="Par Rate (%)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _plot_cpr_curve(cpr_months, cpr_curve):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cpr_months,
        y=[v * 100 for v in cpr_curve],
        mode="lines",
        name="CPR (avg-path)",
        line=dict(color="#e74c3c", width=2.5),
    ))
    fig.update_layout(
        title="CPR by Month (Hull-White Average-Rate Path)",
        xaxis_title="Month",
        yaxis_title="CPR (%)",
        height=360,
        margin=dict(l=40, r=20, t=45, b=40),
        hovermode="x unified",
    )
    return fig


def main():
    st.set_page_config(page_title="OAS Calculator", layout="wide")
    st.title("Option-Adjusted Spread Calculator")
    st.caption("Monte Carlo simulation with Hull-White 1-Factor Model")

    loaded_models = _fetch_loaded_models() or AVAILABLE_MODELS
    default_idx = loaded_models.index("xgb") if "xgb" in loaded_models else 0

    curve_date, default_tenors, default_rates = _fetch_latest_par_curve()

    # Sidebar
    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Prepayment model",
            loaded_models,
            index=default_idx,
            help="ML model",
        )

        st.header("Loan / MBS")
        coupon = st.slider("Coupon Rate (%)", 1.0, 15.0, 6.5, 0.125) / 100
        orig_term = st.selectbox("Original Term (months)", [120, 180, 240, 300, 360], index=1)
        orig_balance = st.number_input("Original Balance ($)", 10_000, 5_000_000,
                                       300_000, step=10_000)
        loan_age = st.slider("Loan Age (months)", 0, orig_term - 1, 0)
        current_balance = st.number_input(
            "Current Balance ($)", 0, 5_000_000,
            int(orig_balance), step=10_000,
            help="Leave equal to original balance for a new loan.")

        st.header("Market Price")
        market_price = st.slider("Price (per $100)", 80.0, 120.0, 100.0, 0.25)

        st.header("Simulation")
        n_paths = st.select_slider("MC Paths", [100, 200, 500, 1000, 2000, 5000],
                                   value=100)
        seed = st.number_input("Random Seed", 0, 9999, 42)

    # loan features
    with st.expander("Borrower & loan features (ML model inputs)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            fico = st.slider("FICO", 300, 850, 700, 5)
            orig_ltv = st.slider("LTV (%)", 1, 200, 80, 1)
            dti = st.slider("Debt-to-income ratio (%)", 0, 100, 35, 1)
            origination_year = st.number_input("Origination Year", 1990, 2030, 2020, 1)
            ph_delinq_count = st.number_input("Prior Delinquency Count", 0, 50, 0, 1)
            excess_principal = st.number_input("Excess Principal", 0.0, 1e6, 0.0, 1000.0)
        with c2:
            channel = st.selectbox("Channel", ["R", "B", "C"], index=0,
                                   help="R=Retail, B=Broker, C=Correspondent")
            loan_purpose = st.selectbox("Loan Purpose", ["P", "C", "R", "U"], index=0,
                                        help="P=Purchase, C=Cash-out, R=Refi, U=Unknown")
            property_type = st.selectbox("Property Type",
                                         ["SF", "CO", "PU", "MH", "CP"], index=0,
                                         help="SF=Single-Family, CO=Condominium, PU=Urban Dev, MH= Manufactured, CP=Co-operative")
            occupancy_status = st.selectbox("Occupancy", ["P", "S", "I"], index=0,
                                            help="P=Primary, S=Second, I=Investor")
            property_state = st.selectbox("State", US_STATES,
                                          index=US_STATES.index("CA"))
            hltv_refi_option = st.selectbox("High LTV Refi Option", ["N", "Y"], index=0)
        with c3:
            first_time_buyer = st.checkbox("First-time buyer", value=False)
            modified = st.checkbox("Modified", value=False)
            in_forbearance = st.checkbox("In forbearance", value=False)
            has_deferral = st.checkbox("Has deferral", value=False)
            has_ppm = st.checkbox("Has prepayment penalty", value=False)
            is_io = st.checkbox("Interest only", value=False)
            is_high_bal = st.checkbox("High balance conforming", value=False)

    # Yield Curve
    curve_label = "Yield Curve (edit tenors/rates)"
    if curve_date:
        curve_label += f" — US Treasury par, {curve_date}"
    with st.expander(curve_label, expanded=False):
        if curve_date:
            st.caption(
                f"Pre-filled from US Treasury par yield curve ({curve_date})."
            )
        else:
            st.caption(
                "Treasury feed unavailable"
            )
        col1, col2 = st.columns(2)
        with col1:
            tenors_str = st.text_input(
                "Tenors, years",
                ", ".join(_fmt_tenor(t) for t in default_tenors),
            )
        with col2:
            rates_str = st.text_input(
                "Rates",
                ", ".join(_fmt_rate(r) for r in default_rates),
            )
        tenors = [float(x.strip()) for x in tenors_str.split(",")]
        rates = [float(x.strip()) for x in rates_str.split(",")]

    # request
    payload = {
        "loan": {
            "coupon": coupon,
            "orig_term": orig_term,
            "orig_balance": float(orig_balance),
            "loan_age": loan_age,
            "current_balance": float(current_balance) if current_balance else None,
        },
        "loan_features": {
            "fico": float(fico),
            "orig_ltv": float(orig_ltv),
            "dti": float(dti),
            "channel": channel,
            "loan_purpose": loan_purpose,
            "property_type": property_type,
            "occupancy_status": occupancy_status,
            "property_state": property_state,
            "origination_year": int(origination_year),
            "first_time_buyer": int(first_time_buyer),
            "modified": int(modified),
            "in_forbearance": int(in_forbearance),
            "has_deferral": int(has_deferral),
            "has_ppm": int(has_ppm),
            "is_io": int(is_io),
            "is_high_bal": int(is_high_bal),
            "hltv_refi_option": hltv_refi_option,
            "ph_delinq_count": int(ph_delinq_count),
            "excess_principal": float(excess_principal),
        },
        "yield_curve": {"tenors": tenors, "rates": rates},
        "model_name": model_name,
        "market_price": market_price,
        "n_paths": n_paths,
        "seed": seed,
    }

    # COMPUTE
    if st.button("Compute OAS", type="primary", use_container_width=True):
        with st.spinner(f"Running Monte Carlo simulation using '{model_name}'..."):
            result = _try_api(payload)

        if result is None:
            st.warning("Computation failed. Check that the backend is running.")
            return

        st.success("Computation done")

        r1, r2, r3 = st.columns(3)
        r1.metric("OAS", f"{result['oas_bps']:.1f} bp")
        r2.metric("OAS expected", f"{result['oas_expected_bps']:.1f} bp")
        r3.metric("OAS unexpected", f"{result['oas_unexpected_bps']:.1f} bp")

        r4, r5, r6 = st.columns(3)
        r4.metric("WAL", f"{result['avg_life']:.1f} yrs")
        r5.metric("Avg SMM", f"{result['avg_smm'] * 100:.4f}%")
        r6.metric("CPR", f"{result['avg_cpr'] * 100:.2f}%")

        st.subheader("Visualisations")

        # CPR chart
        if result.get("cpr_curve_monthly"):
            st.plotly_chart(
                _plot_cpr_curve(
                    result.get("cpr_months", []),
                    result.get("cpr_curve_monthly", []),
                ),
                use_container_width=True,
            )

        # Rate paths
        if result.get("rate_paths"):
            st.plotly_chart(
                _plot_rate_paths(
                    result["path_times"],
                    result["rate_paths"],
                    result.get("rate_mean", []),
                    result.get("rate_p05", []),
                    result.get("rate_p95", []),
                ),
                use_container_width=True,
            )

        # Yield curve chart
        st.plotly_chart(_plot_yield_curve(tenors, rates),
                        use_container_width=True)


if __name__ == "__main__":
    main()
