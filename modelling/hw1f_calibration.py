from datetime import datetime
from io import StringIO
import numpy as np
import pandas as pd
import requests

from backend.treasury_curve import CSV_URL


def get_short_rate(years_back):
    this_year = datetime.now().year
    pieces = []

    for year in range(this_year - years_back, this_year + 1):
        url = CSV_URL.format(year=year)
        try:
            text = requests.get(url, timeout=20,
                                headers={"User-Agent": "MyAgent"}).text
            table = pd.read_csv(StringIO(text))
            pieces.append(table[["Date", "1 Mo"]])
        except Exception as error:
            print("Could not download year", year, ":", error)

    data = pd.concat(pieces)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna().sort_values("Date")
    rate = data.set_index("Date")["1 Mo"] / 100.0
    start = rate.index.max() - pd.DateOffset(years=years_back)
    return rate[rate.index >= start]


def fit_parameters(rate):
    dt = 1.0 / 12.0
    monthly = rate.resample("ME").last().dropna()

    r_now = monthly.values[:-1]
    r_next = monthly.values[1:]

    slope, intercept = np.polyfit(r_now, r_next, 1)

    a = (1 - slope) / dt
    b = intercept / (1 - slope)

    noise = r_next - (slope * r_now + intercept)
    sigma = np.std(noise) / np.sqrt(dt)

    return a, sigma, b, len(monthly)


def main():
    print("Fitting Hull-White parameters from the 1-month Treasury rate")
    print("=" * 60)

    for years in [5, 10, 15]:
        rate = get_short_rate(years)
        a, sigma, b, n_months = fit_parameters(rate)

        print()
        print("Last", years, "years  (", n_months, "monthly points )")
        print("  a     (return to average) =", round(a, 4))
        print("  sigma (volatility)        =", round(sigma, 5))
        print("  b     (long term average) =", round(b, 4),
              " = ", round(b * 100, 2), "%")


if __name__ == "__main__":
    main()
