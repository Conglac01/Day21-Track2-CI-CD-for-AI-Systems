from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")


def load_model():
    """
    Load model tu filesystem local de tranh phu thuoc cloud runtime.
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server. GitHub Actions dung endpoint nay de xac nhan deploy thanh cong."""
    # TODO 2.6.4: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.

    Dau vao: JSON {"features": [f1, f2, ..., f12]}
    Dau ra:  JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    # TODO 2.6.5: Kiem tra len(req.features) == 12.
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # TODO 2.6.6: Goi model.predict([req.features]) de lay ket qua du doan.
    pred_int = int(model.predict([req.features])[0])

    # TODO 2.6.7: Tra ve dict chua "prediction" (int) va "label" (string).
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred_int, "label": labels[pred_int]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
