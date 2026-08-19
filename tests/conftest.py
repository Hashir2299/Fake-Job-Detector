import pandas as pd
import pytest

from src.constants import KEEP_COLUMNS


@pytest.fixture
def sample_raw_data_path(tmp_path):
    rows = []
    for index in range(20):
        rows.append(
            {
                "title": f"Software Engineer {index}",
                "description": "Build reliable python services with tests",
                "requirements": "python api testing git",
                "company_profile": "Established technology company",
                "salary_range": "70000-90000",
                "required_experience": "Mid-Senior level",
                "fraudulent": 0,
            }
        )
        rows.append(
            {
                "title": f"Remote Data Entry {index}",
                "description": "Quick money wire transfer processing role",
                "requirements": "no experience required immediate start",
                "company_profile": "New international recruitment team",
                "salary_range": "100000-150000",
                "required_experience": "Entry level",
                "fraudulent": 1,
            }
        )

    path = tmp_path / "fake_job_postings.csv"
    pd.DataFrame(rows, columns=KEEP_COLUMNS).to_csv(path, index=False)
    return path
