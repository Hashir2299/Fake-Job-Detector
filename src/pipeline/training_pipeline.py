import logging

import src.logger
from src.components.data_transformation import prepare_data
from src.components.model_evaluation import evaluate_model
from src.components.model_trainer import train_model

logger = logging.getLogger(__name__)


def run_training_pipeline():
    logger.info("Training pipeline started")
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data()

    model = train_model(X_train, y_train)
    validation_metrics = evaluate_model(model, X_validation, y_validation)
    test_metrics = evaluate_model(model, X_test, y_test)
    logger.info("Training pipeline finished")

    return {
        "validation": validation_metrics,
        "test": test_metrics,
    }


if __name__ == "__main__":
    run_training_pipeline()
