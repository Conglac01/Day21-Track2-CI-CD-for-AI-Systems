from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

AWS_BUCKET = os.environ["AWS_BUCKET"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu S3 ve may khi server khoi dong.
    """
    # TODO 2.6.1: Tao S3 client
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # TODO 2.6.2: Tai file model tu S3
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3.download_file(AWS_BUCKET, MODEL_KEY, MODEL_PATH)

    # TODO 2.6.3: In thong bao thanh cong
    print("Model da duoc tai xuong tu S3.")


# Goi ham nay khi module duoc import (chay khi server khoi dong)
download_model()
model = joblib.load(MODEL_PATH)


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
