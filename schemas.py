from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field

class RouteSchema(BaseModel):
    next_action: Literal[
        "FinancialMetricsAgent",
        "NewsSentimentAgent",
        "MarketIntelligenceAgent",
        "SQLAgent",
        "Reflection",
        "Synthesizer",
    ] = Field(
        ...,
        title="Next",
        description="Select the next role",
    )

