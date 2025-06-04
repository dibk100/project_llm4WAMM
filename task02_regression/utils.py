import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    return {'mse': mse, 'rmse': rmse, 'r2': r2}
