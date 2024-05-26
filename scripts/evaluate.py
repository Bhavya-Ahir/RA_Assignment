import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

from utils import load_data, extract_features, get_indices

def plot_results(results_df, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results_df['E'], results_df['Test MAE'], marker='o', label='Test MAE')
    plt.xlabel('E')
    plt.ylabel('Test MAE')
    plt.title(f'{title} - Test MAE')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(results_df['E'], results_df['Test R2'], marker='o', label='Test R2')
    plt.xlabel('E')
    plt.ylabel('Test R2')
    plt.title(f'{title} - Test R2')
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    BASE_PATH = 'data'
    hp_space, train_loss, eval_loss, eval_acc = load_data(BASE_PATH)
    train_indices, validation_indices, test_indices = get_indices()

    models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
    E_values = [5, 10, 20, 30, 60]

    for model_name in models:
        results = []
        for E in E_values:
            model = joblib.load(f'models/{model_name}_E{E}.pkl')

            test_features = extract_features(hp_space, train_loss, eval_loss, eval_acc, test_indices, E)
            test_targets = eval_acc.iloc[test_indices, 4 + 149].values

            test_predictions = model.predict(test_features)
            test_mae = mean_absolute_error(test_targets, test_predictions)
            test_r2 = r2_score(test_targets, test_predictions)

            results.append({'E': E, 'Test MAE': test_mae, 'Test R2': test_r2})

        results_df = pd.DataFrame(results)
        plot_results(results_df, f'{model_name} Results')
        results_df.to_csv(f'models/{model_name}_test_results.csv', index=False)

if __name__ == "__main__":
    main()