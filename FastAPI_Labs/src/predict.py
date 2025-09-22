import joblib

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted target values
    """
    model = joblib.load("../model/housing_model.pkl")
    y_pred = model.predict(X)
    return y_pred
