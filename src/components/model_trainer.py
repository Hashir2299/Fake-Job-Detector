import logging

import joblib

from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.constants import MODEL_CONFIG_PATH, MODEL_PATH, RANDOM_STATE
from src.utils.main_utils import combine_text_columns, read_yaml

logger = logging.getLogger(__name__)


def train_model(X_train, y_train, model_path=MODEL_PATH):
    config = read_yaml(MODEL_CONFIG_PATH)
    tfidf_config = config.get("tfidf", {})
    lr_config = config.get("logistic_regression", {})

    X_train_text = combine_text_columns(X_train).to_frame(name="job_text")

    sampler = RandomOverSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(
        X_train_text,
        y_train,
    )
    logger.info("Training data oversampled from %s to %s", len(y_train), len(y_resampled))

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=tfidf_config.get("max_features", 50000),
                ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
                min_df=tfidf_config.get("min_df", 2),
                sublinear_tf=tfidf_config.get("sublinear_tf", True),
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=lr_config.get("max_iter", 1000),
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    logger.info("Training model")
    model.fit(X_resampled["job_text"], y_resampled)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    return model
