import logging

import pandas as pd

from src.constants import MODEL_CONFIG_PATH, MODEL_PATH, TEXT_COLUMNS
from src.utils.main_utils import combine_text_columns, load_object, read_yaml

logger = logging.getLogger(__name__)


class FakeJobPredictor:
    def __init__(self, model_path=MODEL_PATH):
        config = read_yaml(MODEL_CONFIG_PATH)
        self.threshold = config.get("logistic_regression", {}).get("threshold", 0.5)
        self.model = load_object(model_path)
        logger.info("Prediction model loaded from %s", model_path)

    def predict(self, job_data: dict) -> dict:
        row = {column: job_data.get(column, "") or "" for column in TEXT_COLUMNS}
        frame = pd.DataFrame([row])
        text = combine_text_columns(frame)

        probability = float(self.model.predict_proba(text)[0][1])
        prediction = int(probability >= self.threshold)
        logger.info("Prediction completed with fake_probability=%s", round(probability, 4))

        return {
            "prediction": prediction,
            "label": "fake" if prediction == 1 else "real",
            "fake_probability": round(probability, 4),
        }
