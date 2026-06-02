from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


Stage = Literal["input", "output"]
Scenario = Literal["text2image", "img2img_stylization", "img2img_editing", "output"]
Verdict = Literal["allow", "review", "block"]
SignalStatus = Literal["ok", "skipped", "error"]
# Роль сигнала в конвейере:
#   "sensor" — независимый детектор, его оценки участвуют в фьюжне (max);
#   "judge"  — «арбитр», чьё мнение используется как подтверждение (gate)
#              для soft-категорий. DecisionEngine различает их по этому полю,
#              а не по имени сенсора.
SignalRole = Literal["sensor", "judge"]


class ModerationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    scenario: Scenario
    stage: Stage = "input"
    prompt: str | None = None
    image_path: str | None = None
    image_base64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalResult(BaseModel):
    name: str
    status: SignalStatus
    role: SignalRole = "sensor"
    categories: dict[str, float] = Field(default_factory=dict)
    text: list[str] = Field(default_factory=list)
    reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ModerationResponse(BaseModel):
    request_id: str
    scenario: Scenario
    stage: Stage
    verdict: Verdict
    categories: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    signals: list[SignalResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

