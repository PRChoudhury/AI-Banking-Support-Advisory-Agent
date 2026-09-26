# Import Third-party libraries
from langchain_core.tools import tool

MONTHS_PER_YEAR = 12
MAX_TENURE_MONTHS = 600  # 50 years - anything beyond this is a bad input, not a loan.


def compute_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """
    Core EMI maths, kept free of the tool decorator so tests can call it directly.

    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1), where r is the monthly rate as a
    fraction and n the number of instalments. A zero-interest loan degenerates
    to a plain P / n, which the formula above cannot express (0 / 0).
    """
    print(f"[calculate_emi CALLED] principal={principal}, rate={annual_rate}, tenure={tenure_months}")
    if principal <= 0:
        raise ValueError("principal must be greater than 0.")
    if annual_rate < 0:
        raise ValueError("annual_rate cannot be negative.")
    if tenure_months <= 0:
        raise ValueError("tenure_months must be at least 1.")
    if tenure_months > MAX_TENURE_MONTHS:
        raise ValueError(f"tenure_months cannot exceed {MAX_TENURE_MONTHS}.")

    monthly_rate = annual_rate / (MONTHS_PER_YEAR * 100)

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        growth = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * growth / (growth - 1)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    return {
        "principal": round(principal, 2),
        "annual_rate_percent": annual_rate,
        "tenure_months": tenure_months,
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2),
    }


def amortization_schedule(principal: float, annual_rate: float, tenure_months: int) -> list[dict]:
    """
    Month-by-month split of every instalment into interest and principal.

    The closing instalment absorbs the rounding drift of all the earlier ones so
    the balance lands exactly on zero rather than a few paise either side.
    """
    emi = compute_emi(principal, annual_rate, tenure_months)["monthly_emi"]
    monthly_rate = annual_rate / (MONTHS_PER_YEAR * 100)

    schedule = []
    balance = principal
    for month in range(1, tenure_months + 1):
        interest = balance * monthly_rate
        repayment = emi - interest
        if month == tenure_months:
            repayment = balance
            emi_due = balance + interest
        else:
            emi_due = emi
        balance -= repayment
        schedule.append({
            "month": month,
            "emi": round(emi_due, 2),
            "interest": round(interest, 2),
            "principal": round(repayment, 2),
            "balance": round(max(balance, 0.0), 2),
        })
    return schedule


@tool("emi_calculator")
def emi_calculator(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """
    Calculate the monthly EMI, total interest and total payable for a loan.

    Use this whenever a customer asks what their instalment would be, how much
    interest a loan costs, or how changing the amount, rate or tenure affects
    the repayment.

    Args:
        principal: Loan amount borrowed, in rupees.
        annual_rate: Annual interest rate as a percentage, e.g. 9.5 for 9.5%.
        tenure_months: Loan tenure in months (multiply years by 12).

    Returns the EMI breakdown, or an ``error`` key when the inputs are invalid.
    """
    try:
        return compute_emi(principal, annual_rate, tenure_months)
    except ValueError as exc:
        return {"error": str(exc)}
