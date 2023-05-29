"""Evaluation script for ML models on test partition."""
import argparse
import logging
import math

from sklearn.metrics import r2_score, mean_squared_error

from src.utils import get_data, load_file

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Evaluating the ML models.")
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="The name of the model to be loaded.",
    )
    parser.add_argument(
        "--preprocessor_file",
        type=str,
        default="preprocessor.joblib",
        help="The name of the preprocessor to be loaded.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    logger.info("Arguments are obtained.")

    *_, X_test, y_test = get_data()
    logger.info("Data is obtained.")

    preprocessor = load_file(args.preprocessor_file)
    logger.info("Preprocessor is loaded.")
    model = load_file(args.model_file)
    logger.info("Model is loaded.")

    preprocessed_X_test = preprocessor.transform(X_test)
    prediction = model.predict(preprocessed_X_test)

    # Evaluate predictions
    r2_result = r2_score(y_test, prediction)
    rmse_result = math.sqrt(mean_squared_error(y_test, prediction))
    logger.info("The model performance on the test set:")
    logger.info(f"R2 score: {r2_result:.4}, RMSE: {rmse_result:.2f}")


if __name__ == "__main__":
    main()
