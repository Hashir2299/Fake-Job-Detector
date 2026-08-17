import logging

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import METRICS_PATH
from src.utils.main_utils import combine_text_columns, save_json

logger = logging.getLogger(__name__)


def evaluate_model(model, X_validation, y_validation, metrics_path=METRICS_PATH):
    logger.info("Evaluating model")
    X = combine_text_columns(X_validation)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(
        y_validation,
        predictions,
    )

    metrics = {
        "accuracy": round(accuracy_score(y_validation, predictions), 4),
        "precision": round(precision_score(y_validation, predictions), 4),
        "recall": round(recall_score(y_validation, predictions), 4),
        "f1": round(f1_score(y_validation, predictions), 4),
        "pr_auc": round(average_precision_score(y_validation, probabilities), 4),
        "confusion_matrix": cm.tolist(),
    }

    save_json(metrics, metrics_path)
    logger.info("Metrics saved to %s", metrics_path)

    for name, value in metrics.items():
        logger.info("%s: %s", name, value)

    return metrics
