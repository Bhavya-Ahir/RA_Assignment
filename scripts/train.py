import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from utils import load_data, extract_features, get_indices

def train_and_evaluate_model(model, E, hp_space, train_loss, eval_loss, eval_acc, train_indices, validation_indices, test_indices):
    train_features = extract_features(hp_space, train_loss, eval_loss, eval_acc, train_indices, E)
    validation_features = extract_features(hp_space, train_loss, eval_loss, eval_acc, validation_indices, E)
    test_features = extract_features(hp_space, train_loss, eval_loss, eval_acc, test_indices, E)

    train_targets = eval_acc.iloc[train_indices, 4 + 149].values
    validation_targets = eval_acc.iloc[validation_indices, 4 + 149].values
    test_targets = eval_acc.iloc[test_indices, 4 + 149].values

    model.fit(train_features, train_targets)

    validation_predictions = model.predict(validation_features)
    validation_mae = mean_absolute_error(validation_targets, validation_predictions)
    validation_r2 = r2_score(validation_targets, validation_predictions)

    test_predictions = model.predict(test_features)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_r2 = r2_score(test_targets, test_predictions)

    return (E, validation_mae, validation_r2, test_mae, test_r2)

def main():
    BASE_PATH = 'data'
    hp_space, train_loss, eval_loss, eval_acc = load_data(BASE_PATH)
    train_indices, validation_indices, test_indices = get_indices()

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    E_values = [5, 10, 20, 30, 60]
    results = {model_name: [] for model_name in models.keys()}

    for E in E_values:
        for model_name, model in models.items():
            result = train_and_evaluate_model(model, E, hp_space, train_loss, eval_loss, eval_acc, train_indices, validation_indices, test_indices)
            results[model_name].append(result)
            joblib.dump(model, f'models/{model_name}_E{E}.pkl')

    for model_name, model_results in results.items():
        df = pd.DataFrame(model_results, columns=['E', 'Validation MAE', 'Validation R2', 'Test MAE', 'Test R2'])
        df.to_csv(f'models/{model_name}_results.csv', index=False)

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    main()
