from typing import Any, Dict, Literal, Optional
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


class BenchmarkRecord(BaseModel):
    question_id: int
    attempt: int
    temperature: float
    validation_passed: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    ttft: float
    total_time: float
    tokens_per_second: float
    ollama_ram_mb: float
    system_ram_percent: float
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
