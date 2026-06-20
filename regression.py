import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_local_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use CSV, Excel, or Parquet.")


def run_regression(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> None:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataset.")

    cleaned = df.dropna().copy()
    y = cleaned[target_column]
    X = cleaned.drop(columns=[target_column])

    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    pearson_r, pearson_p = stats.pearsonr(y_test, predictions)

    print("Linear Regression Results")
    print(f"Rows used: {len(cleaned)}")
    print(f"Features: {X.shape[1]}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")
    print(f"Pearson r: {pearson_r:.4f} (p-value: {pearson_p:.4g})")

    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    print("\nStatsmodels OLS Summary")
    print(ols_model.summary())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple linear regression on a local dataset.")
    parser.add_argument("dataset_path", help="Path to a local CSV, Excel, or Parquet dataset.")
    parser.add_argument("target_column", help="Name of the target column to predict.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_local_dataset(args.dataset_path)
    run_regression(df, args.target_column, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
