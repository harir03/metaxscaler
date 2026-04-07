from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Decision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    CONDITIONAL = "conditional"


class CreditAction(BaseModel):
    decision: str = Field(..., description="approve, reject, or conditional")
    reasoning: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_terms: Optional[str] = None


class CompanyProfile(BaseModel):
    name: str
    sector: str
    incorporation_year: int
    annual_turnover_cr: float
    loan_type: str
    loan_amount_cr: float


class FinancialMetrics(BaseModel):
    dscr: float
    current_ratio: float
    debt_equity_ratio: float
    net_profit_margin: float
    revenue_growth_yoy: float
    interest_coverage_ratio: float
    working_capital_days: int
    cash_flow_positive: bool


class RiskIndicators(BaseModel):
    credit_rating: str
    wilful_defaulter: bool = False
    active_criminal_case: bool = False
    nclt_active: bool = False
    gst_compliance_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    related_party_transactions_flagged: bool = False
    audit_qualifications: int = 0
    promoter_pledge_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class MarketContext(BaseModel):
    sector_outlook: str   # positive / neutral / negative
    sector_npa_rate: float
    gdp_growth_relevant: float
    regulatory_risk: str  # low / medium / high


class CreditObservation(BaseModel):
    company: CompanyProfile
    financials: FinancialMetrics
    risk: RiskIndicators
    market: MarketContext
    step: int
    task_name: str
    difficulty: str
    hint: Optional[str] = None


class CreditState(BaseModel):
    episode_id: str
    step: int = 0
    task_name: str = ""
    difficulty: str = "easy"
    done: bool = False
    ground_truth_decision: str = ""
    ground_truth_score: int = 0
    company_name: str = ""
    max_steps: int = 1


class EnvResult(BaseModel):
    observation: CreditObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
