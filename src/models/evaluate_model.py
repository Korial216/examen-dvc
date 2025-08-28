import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def main():
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    pd.DataFrame({"y_test": y_test.values.ravel(), "predictions": preds}).to_csv("data/predictions.csv", index=False)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)

if __name__ == "__main__":
    main()
