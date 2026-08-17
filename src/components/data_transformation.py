import logging

from sklearn.model_selection import train_test_split

from src.components.data_ingestion import read_raw_data
from src.constants import (
    KEEP_COLUMNS,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def prepare_data(path=None, save_processed=True):
    df = read_raw_data(path) if path else read_raw_data()
    logger.info("Starting data transformation")

    df = df[KEEP_COLUMNS].copy()
    df = df.dropna(subset=REQUIRED_INPUT_COLUMNS + [TARGET_COLUMN])

    for column in KEEP_COLUMNS:
        if column != TARGET_COLUMN:
            df[column] = df[column].fillna("")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df = df.drop_duplicates()

    if save_processed:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info("Processed data saved to %s", PROCESSED_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(
        "Data split complete: train=%s validation=%s test=%s",
        X_train.shape,
        X_validation.shape,
        X_test.shape,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )
