from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI(title="AgriFlow AI Risk Scoring Engine", version="1.0")

# Validation of Sturucture of Info that will be provided by Backend  (Validation)
class FarmerInput(BaseModel):
    field_size_ha: float
    ndvi: float
    ndwi: float
    soil_moisture_pct: float
    precipitation_30d_mm: float

# Moving to backup
model = xgb.Booster()
model.load_model("model/agriflow_xgboost.json")

def get_recommendation(score: float):
    if score >= 80:
        return "Green Loan adayı"
    elif score >= 60:
        return "Standard rate"
    elif score >= 40:
        return "Manual review"
    else:
        return "Yüksek risk"

@app.post("/predict")
async def predict_field_risk(data: FarmerInput):
    # Gələn datanı modelin oxuya biləcəyi formata salırıq
    input_df = pd.DataFrame([data.dict()])
    dmatrix = xgb.DMatrix(input_df)
    
    # AI Precision
    predicted_score = float(model.predict(dmatrix)[0])
    predicted_score = round(max(0.0, min(100.0, predicted_score)), 1)
    
    recommendation = get_recommendation(predicted_score)
    
    return {
        "status": "success",
        "field_efficiency_score": predicted_score,
        "recommendation": recommendation
    }

