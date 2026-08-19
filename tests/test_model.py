from src.components.data_transformation import prepare_data
from src.components.model_trainer import train_model
from src.utils.main_utils import combine_text_columns


def test_model_can_train_and_predict(tmp_path, sample_raw_data_path):
    X_train, X_validation, _, y_train, _, _ = prepare_data(
        path=sample_raw_data_path,
        save_processed=False,
    )

    train_sample = X_train.groupby(y_train).head(50)
    target_sample = y_train.loc[train_sample.index]

    model = train_model(
        train_sample,
        target_sample,
        model_path=tmp_path / "model.pkl",
    )

    predictions = model.predict(combine_text_columns(X_validation.head(5)))

    assert len(predictions) == 5
