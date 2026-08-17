import logging

import pandas as pd

from src.constants import KEEP_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame) -> None:
    missing_columns = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")

    target_values = set(df[TARGET_COLUMN].dropna().unique())
    allowed_values = {0, 1}
    if not target_values.issubset(allowed_values):
        raise ValueError(
            f"Target column must contain only 0 and 1. Found: {target_values}"
        )

    logger.info("Data validation passed")
