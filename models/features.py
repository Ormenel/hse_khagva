NUMERIC_FEATURES = [
    "fico", "orig_interest_rate", "current_interest_rate",
    "refi_incentive", "refi_incentive_pos",
    "rate_spread_to_10y", "spread_pos",
    "orig_ltv", "dti", "orig_upb", "upb_fraction", "equity_proxy",
    "loan_age", "age_sq", "burnout", "pct_term_elapsed",
    "orig_loan_term", "remaining_months_to_mat", "rate_duration",
    "burnout_x_refi", "fico_x_refi", "ltv_x_refi",
    "ph_delinq_count", "excess_principal", "gs10_monthly",
    "logit_rate_spread_to_10y",
]

CATEGORICAL_FEATURES = [
    "channel", "loan_purpose", "property_type", "occupancy_status",
    "fico_bucket", "seasoning_bucket", "month_of_year", "vintage_year",
]

BINARY_FEATURES = [
    "high_ltv", "term_15y", "is_refi", "is_cashout", "is_io",
    "has_ppm", "modified", "is_investor", "is_high_bal",
    "first_time_buyer", "in_forbearance", "has_deferral",
    "is_judicial_state", "is_hltv_refi",
]

_JUDICIAL_STATES = [
    "CT", "DE", "FL", "HI", "IL", "IN", "IA", "KS", "KY", "LA",
    "ME", "MD", "MA", "MN", "MO", "MT", "NE", "NJ", "NM", "NY",
    "ND", "OH", "OK", "PA", "RI", "SC", "SD", "VT", "WI",
]
