from typing import Optional

from pydantic import BaseModel, Field


class JobPostRequest(BaseModel):
    title: str = Field(..., min_length=2)
    description: str = Field(..., min_length=10)
    requirements: str = Field(..., min_length=2)
    company_profile: Optional[str] = ""
    salary_range: Optional[str] = ""
    required_experience: Optional[str] = ""


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fake_probability: float
