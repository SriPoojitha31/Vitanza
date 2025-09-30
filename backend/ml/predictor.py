import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class OutbreakRiskPredictor:
    def __init__(self):
        # Use RobustScaler for better handling of outliers
        self.model = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200, 
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                n_jobs=-1
            ))
        ])
        self.is_trained = False
        self.class_weights = None

    def ensure_trained(self):
        if self.is_trained:
            return
        # Generate a larger, more balanced synthetic dataset
        rng = np.random.default_rng(42)
        n = 2000  # Increased dataset size
        
        # Generate more realistic water quality data
        ph = rng.normal(7.0, 1.2, n)
        turb = np.abs(rng.normal(1.5, 1.8, n))
        tds = np.clip(rng.normal(300, 200, n), 50, 2000)
        temp = np.clip(rng.normal(25, 8, n), 10, 50)
        
        # Add some correlation between features
        ph = np.clip(ph, 4.0, 10.0)
        turb = np.clip(turb, 0.1, 15.0)
        
        X = pd.DataFrame({
            'ph': ph,
            'turbidity': turb,
            'tds': tds,
            'temp': temp,
        })
        
        # More sophisticated risk assessment with balanced classes
        risk_factors = (
            (turb > 3.0) * 0.3 +  # Turbidity risk
            (ph < 6.5) * 0.4 +    # Low pH risk
            (ph > 8.5) * 0.4 +    # High pH risk
            (tds > 1000) * 0.2 +  # High TDS risk
            (temp > 35) * 0.1 +   # High temp risk
            ((turb > 2.0) & (ph < 6.8)) * 0.3 +  # Combined risk
            ((turb > 2.0) & (ph > 8.2)) * 0.3 +  # Combined risk
            ((tds > 800) & (turb > 1.5)) * 0.2   # Combined risk
        )
        
        # Create balanced classes (approximately 30% positive cases)
        y = (risk_factors > 0.3).astype(int)
        
        # Ensure we have both classes
        if y.sum() < 100:
            y = (risk_factors > 0.2).astype(int)
        if y.sum() > n * 0.7:
            y = (risk_factors > 0.4).astype(int)
            
        self.train(X, pd.Series(y))

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the outbreak risk prediction model with validation.
        X: DataFrame with features (water quality, health reports, weather data)
        y: Series with target labels (e.g., 0: Low risk, 1: High risk)
        """
        # Compute class weights for balancing
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, class_weights))
        
        # Update model with computed weights
        self.model.named_steps['clf'].class_weight = self.class_weights
        
        # Perform cross-validation to check for overfitting
        cv_scores = cross_val_score(self.model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Print training results
        print(f"Training completed. Cross-validation AUC scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        print(f"Class weights: {self.class_weights}")
        
        # Check for overfitting (high variance in CV scores)
        if cv_scores.std() > 0.1:
            print("WARNING: High variance in CV scores - possible overfitting detected")
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_auc': cv_scores.mean(),
            'std_cv_auc': cv_scores.std(),
            'class_distribution': y.value_counts().to_dict(),
            'class_weights': self.class_weights
        }

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
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            self.ensure_trained()
        
        feature_names = ['ph', 'turbidity', 'tds', 'temp']
        importances = self.model.named_steps['clf'].feature_importances_
        return dict(zip(feature_names, importances))
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model performance with detailed metrics."""
        if not self.is_trained:
            self.ensure_trained()
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': self.get_feature_importance()
        }

# Example usage (to be replaced with actual data loading and preprocessing):
# predictor = OutbreakRiskPredictor()
# predictor.train(X_train, y_train)
# predictions = predictor.predict(X_test)