import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def main():
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    model = RandomForestRegressor(random_state=42)
    params = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    
    grid = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/best_params.pkl", "wb") as f:
        pickle.dump(grid.best_params_, f)

if __name__ == "__main__":
    main()
