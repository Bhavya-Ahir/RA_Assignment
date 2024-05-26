import pandas as pd
import numpy as np

def load_data(base_path):
    hp_space = pd.read_csv(base_path + '/HP_space.csv')
    train_loss = pd.read_csv(base_path +'/train_loss.csv')
    eval_loss = pd.read_csv(base_path +'/eval_loss.csv')
    eval_acc = pd.read_csv(base_path +'/eval_acc.csv')

    # I noticed there were very few NaN values, I am handling them by forward fill, basically filling them with values from previous row
    hp_space.fillna(method='ffill', inplace=True)
    train_loss.fillna(method='ffill', inplace=True)
    eval_loss.fillna(method='ffill', inplace=True)
    eval_acc.fillna(method='ffill', inplace=True)
    
    return hp_space, train_loss, eval_loss, eval_acc

def extract_features(hp_space, train_loss, eval_loss, eval_acc, indices, E):
    features = []
    for idx in indices:
        hp_values = hp_space.iloc[idx].values
        train_loss_values = train_loss.iloc[idx, 4:(4 + E * 50)].values
        eval_loss_values = eval_loss.iloc[idx, 4:(4 + E)].values
        eval_acc_values = eval_acc.iloc[idx, 4:(4 + E)].values

        # As training data has mini batches I am falttening to match dimensions
        train_loss_values = train_loss_values.reshape(-1)

        feature_vector = np.concatenate([hp_values, train_loss_values, eval_loss_values, eval_acc_values])
        features.append(feature_vector)

    return np.array(features)

def get_indices():
    train_indices = [i-1 for i in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202]]
    validation_indices = [i-1 for i in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199]]
    test_indices = [i-1 for i in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201]]
    return train_indices, validation_indices, test_indices
