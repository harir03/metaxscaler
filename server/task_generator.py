import random
from typing import Tuple

from models import (
    CompanyProfile, FinancialMetrics, RiskIndicators,
    MarketContext, CreditObservation,
)

SECTORS = [
    "Information Technology", "Pharmaceuticals", "Textiles",
    "Steel & Metals", "FMCG", "Real Estate", "Infrastructure",
    "Automobiles", "Chemicals", "Agriculture", "Renewable Energy",
    "Banking & Finance", "Telecom", "Media & Entertainment",
]

LOAN_TYPES = ["Working Capital", "Term Loan", "Project Finance", "Trade Finance"]

_PREFIXES = [
    "Apex", "Global", "National", "Premier", "Stellar",
    "Bharat", "Indo", "Metro", "Pacific", "Crown",
    "Zenith", "Pinnacle", "Titan", "Nova", "Omega",
]
_SUFFIXES = [
    "Industries", "Corp", "Enterprises", "Solutions", "Holdings",
    "Infra", "Tech", "Pharmaceuticals", "Exports", "Group",
]

_GOOD_RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A"]
_MID_RATINGS = ["A-", "BBB+", "BBB", "BBB-"]
_BAD_RATINGS = ["BB+", "BB", "BB-", "B+", "B", "B-", "C", "D"]


def _rand_name():
    return f"{random.choice(_PREFIXES)} {random.choice(_SUFFIXES)} Ltd."


def _make_easy_approve():
    company = CompanyProfile(
        name=_rand_name(), sector=random.choice(SECTORS),
        incorporation_year=random.randint(1985, 2010),
        annual_turnover_cr=round(random.uniform(200, 5000), 2),
        loan_type=random.choice(LOAN_TYPES),
        loan_amount_cr=round(random.uniform(10, 200), 2),
    )
    fin = FinancialMetrics(
        dscr=round(random.uniform(1.8, 3.5), 2),
        current_ratio=round(random.uniform(1.5, 3.0), 2),
        debt_equity_ratio=round(random.uniform(0.2, 0.8), 2),
        net_profit_margin=round(random.uniform(8.0, 25.0), 2),
        revenue_growth_yoy=round(random.uniform(5.0, 30.0), 2),
        interest_coverage_ratio=round(random.uniform(3.0, 8.0), 2),
        working_capital_days=random.randint(30, 90),
        cash_flow_positive=True,
    )
    risk = RiskIndicators(
        credit_rating=random.choice(_GOOD_RATINGS),
        gst_compliance_pct=round(random.uniform(95.0, 100.0), 1),
        promoter_pledge_pct=round(random.uniform(0.0, 10.0), 1),
    )
    mkt = MarketContext(
        sector_outlook=random.choice(["positive", "neutral"]),
        sector_npa_rate=round(random.uniform(1.0, 4.0), 2),
        gdp_growth_relevant=round(random.uniform(5.0, 8.0), 2),
        regulatory_risk="low",
    )
    obs = CreditObservation(
        company=company, financials=fin, risk=risk, market=mkt,
        step=1, task_name="credit-approval-easy", difficulty="easy",
        hint="This company has strong financials and a clean compliance record.",
    )
    return obs, "approve", 720, "Strong financials, good rating, no flags"


def _make_easy_reject():
    company = CompanyProfile(
        name=_rand_name(), sector=random.choice(SECTORS),
        incorporation_year=random.randint(2015, 2024),
        annual_turnover_cr=round(random.uniform(5, 50), 2),
        loan_type=random.choice(LOAN_TYPES),
        loan_amount_cr=round(random.uniform(50, 500), 2),
    )
    fin = FinancialMetrics(
        dscr=round(random.uniform(0.3, 0.9), 2),
        current_ratio=round(random.uniform(0.4, 0.9), 2),
        debt_equity_ratio=round(random.uniform(3.0, 8.0), 2),
        net_profit_margin=round(random.uniform(-15.0, -1.0), 2),
        revenue_growth_yoy=round(random.uniform(-20.0, -3.0), 2),
        interest_coverage_ratio=round(random.uniform(0.3, 0.9), 2),
        working_capital_days=random.randint(180, 365),
        cash_flow_positive=False,
    )
    has_hard_block = random.choice([True, False])
    risk = RiskIndicators(
        credit_rating=random.choice(_BAD_RATINGS),
        wilful_defaulter=has_hard_block,
        active_criminal_case=random.choice([True, False]) if has_hard_block else False,
        nclt_active=random.choice([True, False]),
        gst_compliance_pct=round(random.uniform(30.0, 65.0), 1),
        related_party_transactions_flagged=True,
        audit_qualifications=random.randint(2, 5),
        promoter_pledge_pct=round(random.uniform(60.0, 95.0), 1),
    )
    mkt = MarketContext(
        sector_outlook="negative",
        sector_npa_rate=round(random.uniform(8.0, 18.0), 2),
        gdp_growth_relevant=round(random.uniform(1.0, 3.0), 2),
        regulatory_risk="high",
    )
    obs = CreditObservation(
        company=company, financials=fin, risk=risk, market=mkt,
        step=1, task_name="credit-approval-easy", difficulty="easy",
        hint="This company shows severe financial distress and compliance issues.",
    )
    return obs, "reject", 250, "DSCR < 1, losses, hard blocks present"


def _make_medium():
    company = CompanyProfile(
        name=_rand_name(), sector=random.choice(SECTORS),
        incorporation_year=random.randint(2000, 2018),
        annual_turnover_cr=round(random.uniform(50, 500), 2),
        loan_type=random.choice(LOAN_TYPES),
        loan_amount_cr=round(random.uniform(20, 150), 2),
    )
    if random.random() < 0.5:
        # borderline conditional-approve
        fin = FinancialMetrics(
            dscr=round(random.uniform(1.1, 1.5), 2),
            current_ratio=round(random.uniform(1.0, 1.4), 2),
            debt_equity_ratio=round(random.uniform(1.2, 2.0), 2),
            net_profit_margin=round(random.uniform(3.0, 8.0), 2),
            revenue_growth_yoy=round(random.uniform(-2.0, 10.0), 2),
            interest_coverage_ratio=round(random.uniform(1.5, 2.5), 2),
            working_capital_days=random.randint(90, 150),
            cash_flow_positive=random.choice([True, True, False]),
        )
        risk = RiskIndicators(
            credit_rating=random.choice(_MID_RATINGS),
            gst_compliance_pct=round(random.uniform(75.0, 92.0), 1),
            related_party_transactions_flagged=random.choice([True, False]),
            audit_qualifications=random.randint(0, 1),
            promoter_pledge_pct=round(random.uniform(15.0, 40.0), 1),
        )
        gt_decision, gt_score = "conditional", random.randint(500, 620)
        reason = "Borderline financials, acceptable but needs conditions"
    else:
        # borderline reject
        fin = FinancialMetrics(
            dscr=round(random.uniform(0.9, 1.2), 2),
            current_ratio=round(random.uniform(0.8, 1.2), 2),
            debt_equity_ratio=round(random.uniform(2.0, 3.5), 2),
            net_profit_margin=round(random.uniform(-2.0, 4.0), 2),
            revenue_growth_yoy=round(random.uniform(-8.0, 3.0), 2),
            interest_coverage_ratio=round(random.uniform(0.8, 1.5), 2),
            working_capital_days=random.randint(120, 200),
            cash_flow_positive=random.choice([True, False, False]),
        )
        risk = RiskIndicators(
            credit_rating=random.choice(_MID_RATINGS + _BAD_RATINGS[:2]),
            gst_compliance_pct=round(random.uniform(60.0, 80.0), 1),
            related_party_transactions_flagged=True,
            audit_qualifications=random.randint(1, 3),
            promoter_pledge_pct=round(random.uniform(35.0, 55.0), 1),
        )
        gt_decision, gt_score = "reject", random.randint(350, 500)
        reason = "Weak financials, high leverage, RPT flags"

    mkt = MarketContext(
        sector_outlook=random.choice(["neutral", "negative"]),
        sector_npa_rate=round(random.uniform(4.0, 10.0), 2),
        gdp_growth_relevant=round(random.uniform(3.0, 6.0), 2),
        regulatory_risk=random.choice(["medium", "high"]),
    )
    obs = CreditObservation(
        company=company, financials=fin, risk=risk, market=mkt,
        step=1, task_name="credit-approval-medium", difficulty="medium",
    )
    return obs, gt_decision, gt_score, reason


# --- hard task variants (S3: mixed ground truths) ---

def _make_hard_reject():
    """Profiles that look good on paper but have buried red flags."""
    company = CompanyProfile(
        name=_rand_name(), sector=random.choice(SECTORS),
        incorporation_year=random.randint(1995, 2015),
        annual_turnover_cr=round(random.uniform(100, 2000), 2),
        loan_type=random.choice(LOAN_TYPES),
        loan_amount_cr=round(random.uniform(50, 500), 2),
    )

    trap = random.choice(["wilful", "rev_inflate", "circular", "evergreen"])

    if trap == "wilful":
        fin = FinancialMetrics(
            dscr=round(random.uniform(2.0, 3.0), 2),
            current_ratio=round(random.uniform(1.8, 2.5), 2),
            debt_equity_ratio=round(random.uniform(0.3, 0.7), 2),
            net_profit_margin=round(random.uniform(12.0, 20.0), 2),
            revenue_growth_yoy=round(random.uniform(15.0, 35.0), 2),
            interest_coverage_ratio=round(random.uniform(4.0, 7.0), 2),
            working_capital_days=random.randint(30, 60),
            cash_flow_positive=True,
        )
        risk = RiskIndicators(
            credit_rating=random.choice(_GOOD_RATINGS[:3]),
            wilful_defaulter=True,
            gst_compliance_pct=round(random.uniform(95.0, 100.0), 1),
            promoter_pledge_pct=round(random.uniform(0.0, 5.0), 1),
        )
        reason = "wilful defaulter despite excellent financials"

    elif trap == "rev_inflate":
        # crazy growth but margins are trash and cash flow negative
        fin = FinancialMetrics(
            dscr=round(random.uniform(1.5, 2.5), 2),
            current_ratio=round(random.uniform(1.2, 1.8), 2),
            debt_equity_ratio=round(random.uniform(0.5, 1.2), 2),
            net_profit_margin=round(random.uniform(1.0, 3.0), 2),
            revenue_growth_yoy=round(random.uniform(80.0, 200.0), 2),
            interest_coverage_ratio=round(random.uniform(2.0, 3.0), 2),
            working_capital_days=random.randint(150, 250),
            cash_flow_positive=False,
        )
        risk = RiskIndicators(
            credit_rating=random.choice(_MID_RATINGS),
            gst_compliance_pct=round(random.uniform(70.0, 85.0), 1),
            related_party_transactions_flagged=True,
            audit_qualifications=random.randint(1, 2),
            promoter_pledge_pct=round(random.uniform(30.0, 50.0), 1),
        )
        reason = "revenue inflation — high growth but low margins and negative cash flow"

    elif trap == "circular":
        fin = FinancialMetrics(
            dscr=round(random.uniform(1.3, 2.0), 2),
            current_ratio=round(random.uniform(1.4, 2.2), 2),
            debt_equity_ratio=round(random.uniform(0.8, 1.5), 2),
            net_profit_margin=round(random.uniform(5.0, 10.0), 2),
            revenue_growth_yoy=round(random.uniform(25.0, 50.0), 2),
            interest_coverage_ratio=round(random.uniform(2.0, 4.0), 2),
            working_capital_days=random.randint(10, 25),  # way too low
            cash_flow_positive=True,
        )
        risk = RiskIndicators(
            credit_rating=random.choice(_GOOD_RATINGS),
            related_party_transactions_flagged=True,
            audit_qualifications=random.randint(2, 4),
            promoter_pledge_pct=round(random.uniform(40.0, 70.0), 1),
        )
        reason = "circular trading — RPT flagged, multiple audit quals, high pledge"

    else:  # evergreen
        fin = FinancialMetrics(
            dscr=round(random.uniform(1.0, 1.3), 2),
            current_ratio=round(random.uniform(1.0, 1.3), 2),
            debt_equity_ratio=round(random.uniform(1.5, 2.5), 2),
            net_profit_margin=round(random.uniform(2.0, 5.0), 2),
            revenue_growth_yoy=round(random.uniform(0.0, 3.0), 2),
            interest_coverage_ratio=round(random.uniform(1.0, 1.5), 2),
            working_capital_days=random.randint(100, 180),
            cash_flow_positive=random.choice([True, False]),
        )
        risk = RiskIndicators(
            credit_rating=random.choice(_MID_RATINGS),
            nclt_active=True,
            related_party_transactions_flagged=True,
            audit_qualifications=random.randint(1, 3),
            promoter_pledge_pct=round(random.uniform(50.0, 80.0), 1),
        )
        reason = "evergreening — NCLT active, stagnant growth, high pledge"

    mkt = MarketContext(
        sector_outlook=random.choice(["positive", "neutral"]),
        sector_npa_rate=round(random.uniform(3.0, 8.0), 2),
        gdp_growth_relevant=round(random.uniform(4.0, 7.0), 2),
        regulatory_risk=random.choice(["low", "medium"]),
    )
    obs = CreditObservation(
        company=company, financials=fin, risk=risk, market=mkt,
        step=1, task_name="credit-approval-hard", difficulty="hard",
    )
    return obs, "reject", random.randint(150, 350), reason


def _make_hard_approve():
    """Looks sketchy on the surface but fundamentals are actually solid."""
    company = CompanyProfile(
        name=_rand_name(), sector=random.choice(SECTORS),
        incorporation_year=random.randint(1980, 2005),
        annual_turnover_cr=round(random.uniform(500, 5000), 2),
        loan_type=random.choice(LOAN_TYPES),
        loan_amount_cr=round(random.uniform(30, 200), 2),
    )
    # strong numbers hidden behind yellow flags that aren't dealbreakers
    fin = FinancialMetrics(
        dscr=round(random.uniform(2.0, 3.5), 2),
        current_ratio=round(random.uniform(1.6, 2.8), 2),
        debt_equity_ratio=round(random.uniform(0.3, 0.8), 2),
        net_profit_margin=round(random.uniform(10.0, 20.0), 2),
        revenue_growth_yoy=round(random.uniform(8.0, 25.0), 2),
        interest_coverage_ratio=round(random.uniform(3.5, 7.0), 2),
        working_capital_days=random.randint(40, 80),
        cash_flow_positive=True,
    )
    risk = RiskIndicators(
        credit_rating=random.choice(_MID_RATINGS),  # BBB looks mediocre
        gst_compliance_pct=round(random.uniform(85.0, 93.0), 1),
        related_party_transactions_flagged=True,  # flagged but manageable
        audit_qualifications=1,
        promoter_pledge_pct=round(random.uniform(20.0, 35.0), 1),
    )
    mkt = MarketContext(
        sector_outlook=random.choice(["positive", "neutral"]),
        sector_npa_rate=round(random.uniform(2.0, 5.0), 2),
        gdp_growth_relevant=round(random.uniform(5.0, 7.5), 2),
        regulatory_risk="low",
    )
    obs = CreditObservation(
        company=company, financials=fin, risk=risk, market=mkt,
        step=1, task_name="credit-approval-hard", difficulty="hard",
    )
    return obs, "approve", random.randint(600, 750), \
        "Strong fundamentals despite surface yellow flags — BBB rating but excellent DSCR and margins"


def _make_hard_conditional():
    """Tricky borderline — salvageable with conditions but easy to misjudge."""
    company = CompanyProfile(
        name=_rand_name(), sector=random.choice(SECTORS),
        incorporation_year=random.randint(2005, 2018),
        annual_turnover_cr=round(random.uniform(100, 800), 2),
        loan_type=random.choice(LOAN_TYPES),
        loan_amount_cr=round(random.uniform(40, 250), 2),
    )
    fin = FinancialMetrics(
        dscr=round(random.uniform(1.3, 1.8), 2),
        current_ratio=round(random.uniform(1.2, 1.6), 2),
        debt_equity_ratio=round(random.uniform(1.0, 1.8), 2),
        net_profit_margin=round(random.uniform(5.0, 10.0), 2),
        revenue_growth_yoy=round(random.uniform(10.0, 25.0), 2),
        interest_coverage_ratio=round(random.uniform(2.0, 3.0), 2),
        working_capital_days=random.randint(60, 120),
        cash_flow_positive=True,
    )
    risk = RiskIndicators(
        credit_rating=random.choice(_MID_RATINGS),
        gst_compliance_pct=round(random.uniform(80.0, 92.0), 1),
        related_party_transactions_flagged=True,
        audit_qualifications=random.randint(1, 2),
        promoter_pledge_pct=round(random.uniform(25.0, 45.0), 1),
    )
    mkt = MarketContext(
        sector_outlook="neutral",
        sector_npa_rate=round(random.uniform(4.0, 8.0), 2),
        gdp_growth_relevant=round(random.uniform(4.0, 6.0), 2),
        regulatory_risk="medium",
    )
    obs = CreditObservation(
        company=company, financials=fin, risk=risk, market=mkt,
        step=1, task_name="credit-approval-hard", difficulty="hard",
    )
    return obs, "conditional", random.randint(450, 600), \
        "Decent financials but RPT and pledge concerns need monitoring terms"


def _make_hard():
    r = random.random()
    if r < 0.55:
        return _make_hard_reject()
    elif r < 0.80:
        return _make_hard_approve()
    else:
        return _make_hard_conditional()


def generate_task(task_name: str) -> Tuple[CreditObservation, str, int, str]:
    if task_name == "credit-approval-easy":
        return _make_easy_approve() if random.random() < 0.5 else _make_easy_reject()
    elif task_name == "credit-approval-medium":
        return _make_medium()
    elif task_name == "credit-approval-hard":
        return _make_hard()
    raise ValueError(f"unknown task: {task_name}")
