from pathlib import Path
import logging

import pandas as pd

from src.components.data_validation import validate_data
from src.constants import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def read_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Reading raw data from %s", path)
    df = pd.read_csv(path)
    validate_data(df)
    logger.info("Raw data loaded with shape %s", df.shape)
    return df
