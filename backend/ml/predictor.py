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

    def ensure_trained(self):
        if self.is_trained:
            return
        # Generate a small synthetic dataset for water quality features
        # Features: [ph, turbidity, tds, temp]
        rng = np.random.default_rng(42)
        n = 600
        ph = rng.normal(7.2, 0.8, n)
        turb = np.abs(rng.normal(2.0, 2.0, n))
        tds = np.clip(rng.normal(250, 150, n), 10, 1500)
        temp = np.clip(rng.normal(22, 6, n), 5, 45)
        X = pd.DataFrame({
            'ph': ph,
            'turbidity': turb,
            'tds': tds,
            'temp': temp,
        })
        # More nuanced risk assessment: multiple factors contribute
        y = (
            (turb > 4.0) |  # Lowered threshold
            (ph < 6.0) |    # More strict pH
            (ph > 9.0) |    # More strict pH
            (tds > 800) |   # Higher TDS threshold
            (temp > 40) |   # Higher temp threshold
            ((turb > 2.0) & (ph < 6.5)) |  # Combined factors
            ((turb > 2.0) & (ph > 8.5))    # Combined factors
        ).astype(int)
        self.train(X, pd.Series(y))

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
        self.ensure_trained()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outbreak risk probabilities for new data.
        X: DataFrame with features
        Returns: Array of probabilities for each risk class
        """
        self.ensure_trained()
        return self.model.predict_proba(X)

# Example usage (to be replaced with actual data loading and preprocessing):
# predictor = OutbreakRiskPredictor()
# predictor.train(X_train, y_train)
# predictions = predictor.predict(X_test)