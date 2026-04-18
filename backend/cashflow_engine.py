from dataclasses import dataclass


@dataclass
class LoanParams:
    coupon: float = 0.065
    orig_term: int = 360
    orig_balance: float = 300_000.0
    loan_age: int = 0
    current_balance: float = 0.0

    def __post_init__(self):
        if self.current_balance <= 0:
            self.current_balance = self.orig_balance
