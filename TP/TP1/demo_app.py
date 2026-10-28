from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI() # l’application
class Reading(BaseModel): # le schéma
 temperature_c: float = Field(ge=-40, le=150)
 vibration_mm_s: float = Field(ge=0)
@app.get("/health") # la route GET /health
def health():
 return {"status": "ok"}

@app.post("/v1/predict") # la route POST /v1/predict
def predict(reading: Reading): # reading : déjà validé
 return {"alert": reading.temperature_c > 80 or reading.vibration_mm_s > 5}
