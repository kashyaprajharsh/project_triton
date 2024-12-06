from enum import Enum
from pydantic import BaseModel
from typing import Dict

class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class TimeHorizon(Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class InvestmentStyle(Enum):
    VALUE = "value"
    GROWTH = "growth"
    BLEND = "blend"

class AgentPersonality(BaseModel):
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    investment_style: InvestmentStyle = InvestmentStyle.BLEND
    
    def get_prompt_context(self) -> str:
        return f"""
        Investment Profile:
        - Risk Tolerance: {self.risk_tolerance.value}
        - Time Horizon: {self.time_horizon.value}
        - Investment Style: {self.investment_style.value}
        """