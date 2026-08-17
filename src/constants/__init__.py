from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "fake_job_postings.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "fake_job_postings_clean.csv"
MODEL_PATH = ROOT_DIR / "models" / "fake_job_model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
MODEL_CONFIG_PATH = ROOT_DIR / "config" / "model.yaml"

TARGET_COLUMN = "fraudulent"

KEEP_COLUMNS = [
    "title",
    "description",
    "requirements",
    "company_profile",
    "salary_range",
    "required_experience",
    "fraudulent",
]

REQUIRED_INPUT_COLUMNS = [
    "title",
    "description",
    "requirements",
]

OPTIONAL_INPUT_COLUMNS = [
    "company_profile",
    "salary_range",
    "required_experience",
]

TEXT_COLUMNS = REQUIRED_INPUT_COLUMNS + OPTIONAL_INPUT_COLUMNS

RANDOM_STATE = 42
