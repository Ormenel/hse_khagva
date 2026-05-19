import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import List, Optional, Tuple

import pandas as pd
import requests

log = logging.getLogger(__name__)

CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
    "&page&_format=csv"
)

TENOR_COLS = [
    (0.5,  "6 Mo"),
    (1.0,  "1 Yr"),
    (2.0,  "2 Yr"),
    (3.0,  "3 Yr"),
    (5.0,  "5 Yr"),
    (7.0,  "7 Yr"),
    (10.0, "10 Yr"),
    (20.0, "20 Yr"),
    (30.0, "30 Yr"),
]


@dataclass
class ParCurve:
    date: str
    tenors: List[float]
    rates: List[float]


def fetch_latest_par_curve(timeout: float = 10.0,
                           max_years_back: int = 1) -> ParCurve:
    year = datetime.now(timezone.utc).year
    last_error = None

    for _ in range(max_years_back):
        url = CSV_URL.format(year=year)
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "OAS/1.0"})
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))

            cols = [c for _, c in TENOR_COLS]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"CSV missing columns: {missing}")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False)

            for _, row in df.iterrows():
                vals = row[cols]
                if vals.isna().any():
                    continue
                return ParCurve(
                    date=row["Date"].date().isoformat(),
                    tenors=[t for t, _ in TENOR_COLS],
                    rates=[float(v) / 100.0 for v in vals],
                )
        except Exception as exc:
            last_error = exc
            log.warning("Treasury CSV fetch failed for %s: %s", year, exc)

        year -= 1

    raise RuntimeError(
        f"Could not retrieve a complete Treasury par curve "
        f"(last error: {last_error})"
    )