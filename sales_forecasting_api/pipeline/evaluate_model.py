# evaluate_model.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_model(y_test, y_pred):
    """
    Evaluate the performance of a classification model.
    :param y_test: True labels
    :param y_pred: Predicted labels
    :return: None
    """
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['New Customer', 'Returning Customer'], yticklabels=['New Customer', 'Returning Customer'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True Value')
    plt.show()

def evaluate_regression_model(y_test, y_pred):
    """
    Evaluate the performance of a regression model.
    :param y_test: True values
    :param y_pred: Predicted values
    :return: None
    """
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
