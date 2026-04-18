import numpy as np
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class YieldCurve:
    tenors: np.ndarray = field(default_factory=lambda: np.array([
        0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30
    ]))
    rates: np.ndarray = field(default_factory=lambda: np.array([
        0.045, 0.046, 0.047, 0.048, 0.048, 0.047, 0.046, 0.045, 0.043, 0.042
    ]))

    def __post_init__(self):
        self.tenors = np.asarray(self.tenors, dtype=np.float64)
        self.rates = np.asarray(self.rates, dtype=np.float64)

    def zero_rate(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.tenors, self.rates)

    def discount(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t: np.ndarray, dt: float = 1e-4) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        t_safe = np.maximum(t, 1e-8)
        r_t = self.zero_rate(t_safe)
        r_t_dt = self.zero_rate(t_safe + dt)
        return (r_t_dt * (t_safe + dt) - r_t * t_safe) / dt

    def forward_rate_deriv(self, t: np.ndarray, dt: float = 1e-4) -> np.ndarray:
        f_plus = self.forward_rate(t + dt, dt)
        f_minus = self.forward_rate(t - dt, dt)
        return (f_plus - f_minus) / (2 * dt)


@dataclass
class ParCurveBootstrapper:
    tenors: np.ndarray
    par_rates: np.ndarray
    coupon_freq: int = 2
    face_value: float = 100.0

    def __post_init__(self):
        self.tenors = np.asarray(self.tenors, dtype=np.float64)
        self.par_rates = np.asarray(self.par_rates, dtype=np.float64)

    def bootstrap_zero_curve(self):
        dfs = self.bootstrap_discount_factors()
        zero_rates = -np.log(dfs) / self.tenors
        return YieldCurve(tenors=self.tenors, rates=zero_rates)

    def bootstrap_discount_factors(self) -> np.ndarray:
        df_map = {}

        for T, par_rate in zip(self.tenors, self.par_rates):
            if T <= 1.0 / self.coupon_freq + 1e-12:
                # Single-period approximation:
                # 100 = (100 + c) * DF(T)
                coupon = self.face_value * par_rate / self.coupon_freq
                df_T = self.face_value / (self.face_value + coupon)
                df_map[float(T)] = df_T
                continue

            coupon_dates = self._coupon_dates(T)
            coupon = self.face_value * par_rate / self.coupon_freq

            pv_known = 0.0
            for t in coupon_dates[:-1]:
                df_t = self._get_df(t, df_map)
                pv_known += coupon * df_t

            final_cf = self.face_value + coupon
            df_T = (self.face_value - pv_known) / final_cf

            if df_T <= 0:
                raise ValueError(
                    f"Non-positive discount factor bootstrapped at tenor {T:.6f}. "
                    "Check par rates or tenor structure."
                )

            df_map[float(T)] = df_T

        return np.array([df_map[float(T)] for T in self.tenors], dtype=np.float64)

    def _coupon_dates(self, maturity: float) -> np.ndarray:
        n_payments = int(round(maturity * self.coupon_freq))
        expected_maturity = n_payments / self.coupon_freq

        if not np.isclose(maturity, expected_maturity, atol=1e-10):
            raise ValueError(
                f"Maturity {maturity} is not aligned with coupon frequency "
                f"{self.coupon_freq}. Expected multiples of {1/self.coupon_freq}."
            )

        return np.array(
            [(i + 1) / self.coupon_freq for i in range(n_payments)],
            dtype=np.float64
        )

    def _get_df(self, t: float, df_map: dict) -> float:
        t = float(t)

        if t in df_map:
            return df_map[t]

        known_times = np.array(sorted(df_map.keys()), dtype=np.float64)
        known_dfs = np.array([df_map[x] for x in known_times], dtype=np.float64)

        if len(known_times) == 0:
            raise ValueError(f"No available discount factors to infer DF at t={t}")

        # left extrapolation: use first zero rate
        if t < known_times[0]:
            z0 = -np.log(known_dfs[0]) / known_times[0]
            return float(np.exp(-z0 * t))

        # right extrapolation: use last zero rate
        if t > known_times[-1]:
            z_last = -np.log(known_dfs[-1]) / known_times[-1]
            return float(np.exp(-z_last * t))

        # interpolation inside known range
        log_df = np.interp(t, known_times, np.log(known_dfs))
        return float(np.exp(log_df))


@dataclass
class HullWhiteParams:

    a: float = 0.1 # mean-reversion speed
    sigma: float = 0.01 # short-rate volatility


class HullWhiteModel:

    def __init__(self, params: HullWhiteParams, curve: YieldCurve):
        self.a = params.a
        self.sigma = params.sigma
        self.curve = curve

    def theta(self, t: np.ndarray) -> np.ndarray:
        a, sigma = self.a, self.sigma
        f = self.curve.forward_rate(t)
        f_t = self.curve.forward_rate_deriv(t)
        return f_t + a * f + (sigma ** 2) / (2 * a) * (1 - np.exp(-2 * a * t))

    def simulate(self, n_paths: int, n_steps: int,
                 T: float, r0: Optional[float] = None,
                 seed: int = 42) -> np.ndarray:
        """
        Monte Carlo simulation of short-rate paths via Euler-Maruyama.

        Parameters
        ----------
        n_paths : number of simulated paths
        n_steps : number of time steps per path
        T       : time horizon in years
        r0      : initial short rate (defaults to curve short-end)
        seed    : random seed for reproducibility

        Returns
        -------
        rates : ndarray of shape (n_paths, n_steps+1)
            Simulated short rates at times [0, dt, 2*dt, ..., T].
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)

        if r0 is None:
            r0 = self.curve.forward_rate(np.array([0.0]))[0]

        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = r0

        # Pre-compute theta at each timestep midpoint
        theta_vals = self.theta(t_grid[:-1] + dt / 2)

        # Brownian increments: dW ~ N(0, dt)
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

        for i in range(n_steps):
            r = rates[:, i]
            rates[:, i + 1] = (
                r + (theta_vals[i] - self.a * r) * dt + self.sigma * dW[:, i]
            )

        return rates

    def discount_factors(self, rates: np.ndarray, T: float) -> np.ndarray:
        n_steps = rates.shape[1] - 1
        dt = T / n_steps

        # Cumulative sum of r * dt  → cumulative integral of short rate
        cum_r = np.cumsum(rates[:, :-1] * dt, axis=1)
        # Prepend 1.0 at t=0
        df = np.exp(-np.column_stack([np.zeros(rates.shape[0]), cum_r]))
        return df

    def monthly_discount_factors(self, rates: np.ndarray, T: float) -> np.ndarray:
        return self.discount_factors(rates, T)

"""
tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
par_rates = np.array([0.043, 0.044, 0.045, 0.046, 0.045, 0.044, 0.043, 0.041, 0.040])

bootstrapper = ParCurveBootstrapper(
    tenors=tenors,
    par_rates=par_rates,
    coupon_freq=2
)

curve = bootstrapper.bootstrap_zero_curve()

params = HullWhiteParams(
    a=0.10,      # mean reversion
    sigma=0.01   # short rate vol
)
hw = HullWhiteModel(params, curve)
rates = hw.simulate(
    n_paths=1000,
    n_steps=30 * 12,
    T=30
)
"""
