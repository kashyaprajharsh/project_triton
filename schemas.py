from typing import Literal, Optional, List, Union, Dict , Any
from datetime import datetime
from pydantic import BaseModel, Field


class RouteSchema(BaseModel):
    next_action: Literal[
        "FinancialMetricsAgent",
        "NewsSentimentAgent",
        "MarketIntelligenceAgent",
        "SQLAgent",
        "Reflection",
        "Synthesizer",
        "FINISH"
    ] = Field(
        ...,
        title="Next",
        description="Select the next role",
    )