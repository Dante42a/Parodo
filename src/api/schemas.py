from pydantic import BaseModel, Field


class ToothResult(BaseModel):
    tooth_id: int
    condition: str
    severity: str | None = None
    bone_loss_ratio: float = Field(ge=0, le=1)
    cortical_coverage: float = Field(ge=0, le=1)
    rule_id: str


class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    overall_condition: str
    spread: str | None = None
    overall_severity: str | None = None
    num_affected: int
    reasons: list[str] = Field(default_factory=list)
    rule_version: str
    model_version: str
    processing_ms: float
    teeth: list[ToothResult]
    overlay_png_base64: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    error: str | None = None  # причина, если модель не готова
    warnings: list[str] = Field(default_factory=list)  # модель готова, но с оговорками
