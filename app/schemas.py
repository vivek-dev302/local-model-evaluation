from typing import Literal
from pydantic import BaseModel


class SupportTriageResponse(BaseModel):
    intent: str
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "negative", "frustrated"]
    requires_human: bool
    category: Literal[
        "payment",
        "refund",
        "account",
        "delivery",
        "technical",
        "cancellation",
        "complaint",
        "feature_request",
        "general"
    ]
    suggested_response: str
