from src.components.data_transformation import prepare_data
from src.constants import TEXT_COLUMNS


def test_prepare_data_uses_selected_columns(sample_raw_data_path):
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
        path=sample_raw_data_path,
        save_processed=False
    )

    assert list(X_train.columns) == TEXT_COLUMNS
    assert len(X_train) > 0
    assert len(X_validation) > 0
    assert len(X_test) > 0
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_validation.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
