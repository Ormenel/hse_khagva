"""
frontend.py  –  Streamlit Frontend for OAS Spread Calculator
==============================================================
Interactive dashboard for computing the Option-Adjusted Spread (OAS)
on fixed-rate mortgages / MBS.

Connects to the FastAPI backend (oas_service/api.py) or can be used
standalone (direct import of the calculation engine).

Usage
-----
    # With backend running:
    streamlit run oas_service/frontend.py

    # The frontend calls the FastAPI backend at http://localhost:8000
    # Start the backend first:
    #   uvicorn oas_service.api:app --port 8000 --reload
"""

import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Direct imports for standalone mode (no backend required)
from .hull_white import HullWhiteModel, HullWhiteParams, YieldCurve
from .cashflow_engine import LoanParams, PrepaymentParams, project_cashflows
from .oas_calculator import compute_oas, OASResult


API_URL = "http://localhost:8000"


def _try_api(payload: dict) -> dict | None:
    """Try calling the FastAPI backend; return None if unreachable."""
    try:
        r = requests.post(f"{API_URL}/oas/compute", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
    except requests.ConnectionError:
        pass
    return None


def _compute_local(payload: dict) -> dict:
    """Compute OAS directly without the API (standalone mode)."""
    loan = LoanParams(
        coupon=payload["loan"]["coupon"],
        orig_term=payload["loan"]["orig_term"],
        orig_balance=payload["loan"]["orig_balance"],
        loan_age=payload["loan"]["loan_age"],
        current_balance=(payload["loan"].get("current_balance")
                         or payload["loan"]["orig_balance"]),
    )
    hw = HullWhiteParams(
        a=payload["hull_white"]["a"],
        sigma=payload["hull_white"]["sigma"],
    )
    prepay = PrepaymentParams(
        psa_speed=payload["prepayment"]["psa_speed"],
        refi_sensitivity=payload["prepayment"]["refi_sensitivity"],
        risk_premium=payload["prepayment"]["risk_premium"],
        use_rate_model=payload["prepayment"]["use_rate_model"],
    )
    curve = YieldCurve(
        tenors=np.array(payload["yield_curve"]["tenors"]),
        rates=np.array(payload["yield_curve"]["rates"]),
    )
    result = compute_oas(
        loan=loan, prepay=prepay, curve=curve, hw_params=hw,
        market_price=payload["market_price"],
        n_paths=payload["n_paths"], seed=payload["seed"],
    )
    return {
        "oas_bps": result.oas_bps,
        "model_price": result.model_price,
        "market_price": result.market_price,
        "avg_life": result.avg_life,
        "avg_smm": result.avg_smm,
        "avg_cpr": result.avg_cpr,
        "n_paths": result.n_paths,
        "converged": result.converged,
    }


def _plot_yield_curve(tenors, rates):
    """Plot the user-supplied yield curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenors, y=[r * 100 for r in rates],
        mode="lines+markers", name="Zero Curve",
        line=dict(color="#3498db", width=2),
    ))
    fig.update_layout(
        title="Input Yield Curve",
        xaxis_title="Maturity (years)",
        yaxis_title="Zero Rate (%)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _plot_rate_paths(payload: dict, n_show: int = 30):
    """Simulate and plot a handful of Hull-White rate paths."""
    hw = HullWhiteModel(
        HullWhiteParams(
            a=payload["hull_white"]["a"],
            sigma=payload["hull_white"]["sigma"]),
        YieldCurve(
            tenors=np.array(payload["yield_curve"]["tenors"]),
            rates=np.array(payload["yield_curve"]["rates"])),
    )
    remaining = payload["loan"]["orig_term"] - payload["loan"]["loan_age"]
    T = remaining / 12.0
    rates = hw.simulate(n_show, remaining, T, seed=payload["seed"])
    t_grid = np.linspace(0, T, remaining + 1)

    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=t_grid, y=rates[i] * 100,
            mode="lines", opacity=0.3,
            line=dict(width=1), showlegend=False,
        ))
    # Mean path
    fig.add_trace(go.Scatter(
        x=t_grid, y=np.mean(rates, axis=0) * 100,
        mode="lines", name="Mean Path",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.update_layout(
        title=f"Simulated Short-Rate Paths (n={n_show})",
        xaxis_title="Time (years)",
        yaxis_title="Short Rate (%)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _plot_cashflows(payload: dict):
    """Plot projected cashflows for the mean rate path."""
    hw = HullWhiteModel(
        HullWhiteParams(
            a=payload["hull_white"]["a"],
            sigma=payload["hull_white"]["sigma"]),
        YieldCurve(
            tenors=np.array(payload["yield_curve"]["tenors"]),
            rates=np.array(payload["yield_curve"]["rates"])),
    )
    loan = LoanParams(
        coupon=payload["loan"]["coupon"],
        orig_term=payload["loan"]["orig_term"],
        orig_balance=payload["loan"]["orig_balance"],
        loan_age=payload["loan"]["loan_age"],
        current_balance=(payload["loan"].get("current_balance")
                         or payload["loan"]["orig_balance"]),
    )
    prepay = PrepaymentParams(
        psa_speed=payload["prepayment"]["psa_speed"],
        refi_sensitivity=payload["prepayment"]["refi_sensitivity"],
        risk_premium=payload["prepayment"]["risk_premium"],
        use_rate_model=payload["prepayment"]["use_rate_model"],
    )
    remaining = loan.orig_term - loan.loan_age
    T = remaining / 12.0
    rates = hw.simulate(1, remaining, T, seed=payload["seed"])
    mean_path = rates[0, 1:]

    cf = project_cashflows(loan, prepay, mean_path, 1.0 / 12.0)

    months = np.arange(1, remaining + 1)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Monthly Cash Flows ($)",
                                        "Outstanding Balance & SMM"),
                        vertical_spacing=0.15)
    fig.add_trace(go.Bar(x=months, y=cf["interest"], name="Interest",
                         marker_color="#3498db"), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=cf["scheduled"], name="Sched. Principal",
                         marker_color="#2ecc71"), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=cf["prepayment"], name="Prepayment",
                         marker_color="#e74c3c"), row=1, col=1)

    fig.add_trace(go.Scatter(x=months, y=cf["upb"], name="UPB",
                             line=dict(color="#2c3e50", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=cf["smm"] * 100, name="SMM (%)",
                             line=dict(color="#e67e22", width=2),
                             yaxis="y4"), row=2, col=1)

    fig.update_layout(
        barmode="stack", height=600,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text="Month", row=2, col=1)
    return fig


# ============================================================================
#  STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="OAS Calculator", layout="wide")
    st.title("Option-Adjusted Spread Calculator")
    st.caption("Hull-White 1-Factor Model  |  Monte Carlo Simulation")

    # ── Sidebar: parameters ───────────────────────────────────────────────
    with st.sidebar:
        st.header("Loan / MBS Parameters")
        coupon = st.slider("Coupon Rate (%)", 1.0, 15.0, 6.5, 0.125) / 100
        orig_term = st.selectbox("Original Term (months)", [180, 360], index=1)
        orig_balance = st.number_input("Original Balance ($)", 10_000, 5_000_000,
                                       300_000, step=10_000)
        loan_age = st.slider("Loan Age (months)", 0, orig_term - 1, 0)
        current_balance = st.number_input(
            "Current Balance ($)", 0, 5_000_000,
            int(orig_balance), step=10_000,
            help="Leave equal to original balance for a new loan")

        st.header("Market Price")
        market_price = st.slider("Price (per $100)", 80.0, 120.0, 100.0, 0.25)

        st.header("Hull-White Parameters")
        hw_a = st.slider("Mean Reversion (a)", 0.01, 0.50, 0.10, 0.01)
        hw_sigma = st.slider("Volatility (sigma)", 0.001, 0.050, 0.010, 0.001,
                             format="%.3f")

        st.header("Prepayment Model")
        psa_speed = st.slider("PSA Speed", 0, 500, 150, 10)
        refi_sens = st.slider("Refi Sensitivity", 0.0, 20.0, 5.0, 0.5)
        risk_prem = st.slider("Risk Premium (%)", 0.0, 5.0, 2.0, 0.25) / 100
        use_rate = st.checkbox("Rate-sensitive prepayment", value=True)

        st.header("Simulation")
        n_paths = st.select_slider("MC Paths", [100, 200, 500, 1000, 2000, 5000],
                                   value=500)
        seed = st.number_input("Random Seed", 0, 9999, 42)

    # ── Yield Curve (main area) ───────────────────────────────────────────
    with st.expander("Yield Curve (edit tenors/rates)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            tenors_str = st.text_input(
                "Tenors (years, comma-separated)",
                "0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30")
        with col2:
            rates_str = st.text_input(
                "Rates (decimal, comma-separated)",
                "0.045, 0.046, 0.047, 0.048, 0.048, 0.047, 0.046, 0.045, 0.043, 0.042")
        tenors = [float(x.strip()) for x in tenors_str.split(",")]
        rates = [float(x.strip()) for x in rates_str.split(",")]

    # ── Build request payload ─────────────────────────────────────────────
    payload = {
        "loan": {
            "coupon": coupon,
            "orig_term": orig_term,
            "orig_balance": float(orig_balance),
            "loan_age": loan_age,
            "current_balance": float(current_balance) if current_balance else None,
        },
        "hull_white": {"a": hw_a, "sigma": hw_sigma},
        "prepayment": {
            "psa_speed": float(psa_speed),
            "refi_sensitivity": refi_sens,
            "risk_premium": risk_prem,
            "use_rate_model": use_rate,
        },
        "yield_curve": {"tenors": tenors, "rates": rates},
        "market_price": market_price,
        "n_paths": n_paths,
        "seed": seed,
    }

    # ── Compute ───────────────────────────────────────────────────────────
    if st.button("Compute OAS", type="primary", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            # Try backend first, fall back to direct computation
            result = _try_api(payload)
            mode = "API"
            if result is None:
                result = _compute_local(payload)
                mode = "Local"

        # ── Results ───────────────────────────────────────────────────────
        st.success(f"Computation complete ({mode} mode)")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("OAS", f"{result['oas_bps']:.1f} bp")
        r2.metric("Model Price", f"${result['model_price']:.2f}")
        r3.metric("WAL", f"{result['avg_life']:.1f} yrs")
        r4.metric("CPR", f"{result['avg_cpr'] * 100:.2f}%")

        col_conv, col_smm = st.columns(2)
        col_conv.metric("Converged", "Yes" if result["converged"] else "No")
        col_smm.metric("Avg SMM", f"{result['avg_smm'] * 100:.4f}%")

        # ── Charts ────────────────────────────────────────────────────────
        st.subheader("Visualisations")

        tab1, tab2, tab3 = st.tabs(["Yield Curve", "Rate Paths", "Cash Flows"])

        with tab1:
            st.plotly_chart(_plot_yield_curve(tenors, rates),
                           use_container_width=True)

        with tab2:
            st.plotly_chart(_plot_rate_paths(payload, n_show=30),
                           use_container_width=True)

        with tab3:
            st.plotly_chart(_plot_cashflows(payload),
                           use_container_width=True)


if __name__ == "__main__":
    main()
