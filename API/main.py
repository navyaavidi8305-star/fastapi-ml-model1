from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
app=FastAPI()
model_path=os.path.join(os.path.dirname(__file__),"model.pkl")
model=joblib.load(model_path)
class InputData(BaseModel):
    marks:list[float]
@app.post("/predict")
def predict(data:InputData):
    prediction=model.predict([data.marks])
    return {"prediction":int(prediction[0])}
#running command :- uvicorn main:app --reload  (if we are getting error with this consider the next one)  python -m uvicorn main:app --reload