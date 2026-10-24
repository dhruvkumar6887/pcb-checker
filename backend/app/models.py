from typing import Literal
from pydantic import BaseModel, Field, field_validator

class Component(BaseModel):
    designator: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z][A-Za-z0-9_.-]*$')
    value: str = Field(default='', max_length=256)
    footprint: str = ''
    mpn: str = ''
    part_type: str = ''
    nets: list[str] = Field(default_factory=list)
    roi: list[float] | None = None
    ratios: list[float] | None = None
    baseline_ratios: list[float] | None = None

    @field_validator('roi')
    @classmethod
    def valid_roi(cls, value):
        import math
        if value is not None and (len(value) != 4 or not all(math.isfinite(x) for x in value)
                                  or min(value[:2]) < 0 or min(value[2:]) < 8):
            raise ValueError('ROI must be [x,y,width,height], nonnegative, at least 8 px wide/high.')
        return value

class ComponentsUpdate(BaseModel):
    components: list[Component] = Field(min_length=1, max_length=500)

class Reading(BaseModel):
    measurement: str
    expected: str
    tolerance: str | None = None
    unit: str | None = None
    source: str = 'rules'
    confidence: float = Field(ge=0, le=1)

class ComponentResult(BaseModel):
    designator: str
    value: str
    part_type: str
    status: Literal['present', 'missing', 'uncertain', 'displaced']
    confidence: float = Field(ge=0, le=1)
    roi_px_in_original_test_image: list[float]
    polygon_px_in_original_test_image: list[list[float]]
    roi_canonical: list[float]
    presence_score: float
    signals: dict[str, float]
    probable_defects: list[dict]
    downstream_effect: str
    downstream_effect_source: str = 'rules'
    ideal_reading: Reading
    warnings: list[str]
