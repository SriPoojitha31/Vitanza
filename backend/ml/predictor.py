import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class OutbreakRiskPredictor:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the outbreak risk prediction model.
        X: DataFrame with features (water quality, health reports, weather data)
        y: Series with target labels (e.g., 0: Low risk, 1: High risk)
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outbreak risk for new data.
        X: DataFrame with features
        Returns: Array of predicted risk levels
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outbreak risk probabilities for new data.
        X: DataFrame with features
        Returns: Array of probabilities for each risk class
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        return self.model.predict_proba(X)

# Example usage (to be replaced with actual data loading and preprocessing):
# predictor = OutbreakRiskPredictor()
# predictor.train(X_train, y_train)
# predictions = predictor.predict(X_test)