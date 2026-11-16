"""FastAPI service: uvicorn predict:app --host 0.0.0.0 --port 8000."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommender_santander.columns import PRODUCTS
from recommender_santander.modeling import feature_frame, ranked_indices, score_products


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float | None = Field(default=None, ge=0, le=120)
    tenure_months: float | None = Field(default=None, ge=0)
    household_income: float | None = Field(default=None, ge=0)
    snapshot_month: int = Field(ge=1, le=12)
    sex: str | None = None
    residence_country: str | None = None
    customer_segment: str | None = None
    joining_channel: str | None = None
    is_active_customer: Literal["0", "1"] | None = None
    relationship_month_start: str | None = None


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: CustomerProfile
    owned_products: list[str]  # Complete list: omitted products are treated as unowned.
    k: int = Field(default=7, ge=1, le=7)

    @field_validator("owned_products")
    @classmethod
    def validate_products(cls, products):
        unknown = set(products) - set(PRODUCTS)
        if unknown:
            raise ValueError(f"Unknown products: {sorted(unknown)}")
        if len(products) != len(set(products)):
            raise ValueError("owned_products contains duplicates")
        return products


@asynccontextmanager
async def lifespan(app):
    path = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    if not path.is_file():
        raise RuntimeError(f"Missing model artifact: {path}. Run python train.py first.")
    app.state.bundle = joblib.load(path)
    if "buyer_model" not in app.state.bundle:
        raise RuntimeError("The artifact has no buyer classifier. Run python train.py again.")
    yield


app = FastAPI(title="Santander product recommender", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.bundle["kind"]}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    row = request.profile.model_dump()
    row.update({c: int(c in request.owned_products) for c in PRODUCTS})
    frame = pd.DataFrame([row])
    bundle = app.state.bundle
    buyer_score = float(bundle["buyer_model"].predict_proba(feature_frame(frame))[0, 1])
    should_recommend = buyer_score >= bundle["buyer_threshold"] and len(request.owned_products) < len(PRODUCTS)
    recommendations = []
    if should_recommend:
        scores = score_products(bundle, frame)
        for index in ranked_indices(scores, request.k)[0]:
            if np.isfinite(scores[0, index]):
                recommendations.append({
                    "product": PRODUCTS[index],
                    "ranking_score": round(float(scores[0, index]), 6),
                })

    return {"model": bundle["kind"], "should_recommend": should_recommend,
            "buyer_score": buyer_score, "buyer_threshold": bundle["buyer_threshold"],
            "recommendations": recommendations}
