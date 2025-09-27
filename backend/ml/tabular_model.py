"""
XGBoost Tabular Model for Outbreak Prediction
============================================

This module implements the tabular model using XGBoost for aggregated features.
It handles feature engineering, training, and inference for outbreak prediction.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import os

from schemas.ml_models import TabularFeatures, ModelEvaluation, TrainingData

logger = logging.getLogger(__name__)

class TabularOutbreakPredictor:
    """
    XGBoost-based tabular model for outbreak prediction.
    
    This model uses aggregated features from multiple data sources:
    - Clinical reports (case counts, trends)
    - Environmental data (water quality, weather)
    - Population demographics
    - Geographic features
    - Event flags
    """
    
    Parameters:
    -----------
    model_path : str
        Path to save/load the trained model
    feature_config : dict
        Configuration for feature engineering
    """
    
    def __init__(self, model_path: str = "models/xgb_outbreak.json", 
                 feature_config: Optional[Dict] = None):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_config = feature_config or self._default_feature_config()
        
    def _default_feature_config(self) -> Dict:
        """Default feature engineering configuration."""
        return {
            "temporal_windows": [7, 14, 30],  # Days for rolling averages
            "sensor_aggregations": ["mean", "std", "min", "max"],
            "population_features": ["density", "age_distribution", "sanitation"],
            "environmental_features": ["water_quality", "rainfall", "temperature"],
            "geographic_features": ["latitude", "longitude", "altitude"],
            "event_features": ["festival", "disaster", "outbreak", "contamination"]
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw data with columns for different data sources
            
        Returns:
        --------
        pd.DataFrame
            Engineered features ready for training
        """
        logger.info("Engineering features for tabular model...")
        
        features_df = pd.DataFrame()
        
        # Temporal features - rolling averages and trends
        for window in self.feature_config["temporal_windows"]:
            if 'case_count' in data.columns:
                features_df[f'cases_{window}d_avg'] = data['case_count'].rolling(window=window).mean()
                features_df[f'cases_{window}d_trend'] = data['case_count'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
        
        # Environmental features
        if 'water_quality' in data.columns:
            for agg in self.feature_config["sensor_aggregations"]:
                if agg == "mean":
                    features_df['water_quality_7d_avg'] = data['water_quality'].rolling(7).mean()
                elif agg == "std":
                    features_df['water_quality_7d_std'] = data['water_quality'].rolling(7).std()
        
        if 'rainfall' in data.columns:
            features_df['rainfall_7d_total'] = data['rainfall'].rolling(7).sum()
            features_df['rainfall_30d_total'] = data['rainfall'].rolling(30).sum()
        
        if 'temperature' in data.columns:
            features_df['temperature_7d_avg'] = data['temperature'].rolling(7).mean()
            features_df['temperature_7d_std'] = data['temperature'].rolling(7).std()
        
        # Population features
        if 'population_density' in data.columns:
            features_df['population_density'] = data['population_density']
        
        if 'sanitation_index' in data.columns:
            features_df['sanitation_index'] = data['sanitation_index']
        
        if 'age_median' in data.columns:
            features_df['age_median'] = data['age_median']
        
        # Geographic features
        if 'latitude' in data.columns:
            features_df['latitude'] = data['latitude']
            features_df['longitude'] = data['longitude']
        
        if 'altitude' in data.columns:
            features_df['altitude'] = data['altitude']
        
        # Event flags
        event_columns = ['festival_flag', 'disaster_flag', 'outbreak_flag', 'contamination_flag']
        for col in event_columns:
            if col in data.columns:
                features_df[col] = data[col].astype(int)
        
        # Interaction features
        if 'population_density' in features_df.columns and 'sanitation_index' in features_df.columns:
            features_df['density_sanitation_interaction'] = (
                features_df['population_density'] * features_df['sanitation_index']
            )
        
        if 'water_quality_7d_avg' in features_df.columns and 'rainfall_7d_total' in features_df.columns:
            features_df['water_rainfall_interaction'] = (
                features_df['water_quality_7d_avg'] * features_df['rainfall_7d_total']
            )
        
        # Lag features
        for col in ['case_count', 'water_quality', 'rainfall']:
            if col in data.columns:
                for lag in [1, 3, 7]:
                    features_df[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Seasonal features
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            features_df['month'] = data['date'].dt.month
            features_df['day_of_year'] = data['date'].dt.dayofyear
            features_df['is_monsoon'] = data['date'].dt.month.isin([6, 7, 8, 9]).astype(int)
        
        # Fill missing values
        features_df = features_df.fillna(features_df.median())
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        logger.info(f"Engineered {len(features_df.columns)} features")
        return features_df
    
    def prepare_training_data(self, data: pd.DataFrame, 
                            target_column: str = 'outbreak_flag') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with feature engineering.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw training data
        target_column : str
            Name of the target column
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Features and target labels
        """
        logger.info("Preparing training data...")
        
        # Engineer features
        features_df = self.engineer_features(data)
        
        # Ensure we have the target column
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Align features with target
        features_df = features_df.reindex(data.index)
        target = data[target_column]
        
        # Remove rows with missing target
        valid_mask = ~target.isna()
        features_df = features_df[valid_mask]
        target = target[valid_mask]
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"Prepared {len(features_df)} samples with {len(self.feature_names)} features")
        return features_df, target
    
    def train(self, data: pd.DataFrame, target_column: str = 'outbreak_flag',
              test_size: float = 0.2, validation_size: float = 0.2,
              early_stopping_rounds: int = 20, n_estimators: int = 500) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data for testing
        validation_size : float
            Proportion of data for validation
        early_stopping_rounds : int
            Early stopping rounds
        n_estimators : int
            Number of boosting rounds
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metrics
        """
        logger.info("Training XGBoost tabular model...")
        
        # Prepare data
        X, y = self.prepare_training_data(data, target_column)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + validation_size, 
            stratify=y, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + validation_size),
            stratify=y_temp, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        
        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
            "n_jobs": -1
        }
        
        # Train model
        self.model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50
        )
        
        # Evaluate on test set
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.get_score(importance_type='weight')))
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        
        # Save scaler
        scaler_path = self.model_path.replace('.json', '_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        features_path = self.model_path.replace('.json', '_features.joblib')
        joblib.dump(self.feature_names, features_path)
        
        results = {
            "auc_roc": auc_score,
            "auc_pr": auc_pr,
            "feature_importance": feature_importance,
            "n_features": len(self.feature_names),
            "n_samples": len(X_train),
            "best_iteration": self.model.best_iteration,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Training completed. AUC-ROC: {auc_score:.4f}, AUC-PR: {auc_pr:.4f}")
        return results
    
    def predict(self, features: TabularFeatures) -> float:
        """
        Make prediction for a single sample.
        
        Parameters:
        -----------
        features : TabularFeatures
            Feature values for prediction
            
        Returns:
        --------
        float
            Prediction probability
        """
        if self.model is None:
            self.load_model()
        
        # Convert features to DataFrame
        feature_dict = features.dict()
        feature_df = pd.DataFrame([feature_dict])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(feature_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                feature_df[feature] = 0
        
        # Reorder columns to match training
        feature_df = feature_df[self.feature_names]
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        
        # Make prediction
        dmatrix = xgb.DMatrix(feature_scaled)
        prediction = self.model.predict(dmatrix)[0]
        
        return float(prediction)
    
    def predict_batch(self, features_list: List[TabularFeatures]) -> List[float]:
        """
        Make predictions for multiple samples.
        
        Parameters:
        -----------
        features_list : List[TabularFeatures]
            List of feature sets
            
        Returns:
        --------
        List[float]
            List of prediction probabilities
        """
        if self.model is None:
            self.load_model()
        
        # Convert to DataFrame
        feature_dicts = [features.dict() for features in features_list]
        feature_df = pd.DataFrame(feature_dicts)
        
        # Handle missing features
        missing_features = set(self.feature_names) - set(feature_df.columns)
        for feature in missing_features:
            feature_df[feature] = 0
        
        # Reorder columns
        feature_df = feature_df[self.feature_names]
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        
        # Make predictions
        dmatrix = xgb.DMatrix(feature_scaled)
        predictions = self.model.predict(dmatrix)
        
        return predictions.tolist()
    
    def load_model(self):
        """Load the trained model and preprocessing objects."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        
        # Load scaler
        scaler_path = self.model_path.replace('.json', '_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        features_path = self.model_path.replace('.json', '_features.joblib')
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        
        logger.info("Model loaded successfully")
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Parameters:
        -----------
        importance_type : str
            Type of importance ('weight', 'gain', 'cover')
            
        Returns:
        --------
        Dict[str, float]
            Feature importance scores
        """
        if self.model is None:
            self.load_model()
        
        importance = self.model.get_score(importance_type=importance_type)
        return importance
    
    def cross_validate(self, data: pd.DataFrame, target_column: str = 'outbreak_flag',
                      cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the training data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, float]
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Prepare data
        X, y = self.prepare_training_data(data, target_column)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
            "n_jobs": -1
        }
        
        scores = []
        for train_idx, val_idx in cv.split(X_scaled, y):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
            model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
            y_pred = model.predict(dval)
            score = roc_auc_score(y_val_fold, y_pred)
            scores.append(score)
        
        results = {
            "mean_cv_score": np.mean(scores),
            "std_cv_score": np.std(scores),
            "cv_scores": scores
        }
        
        logger.info(f"Cross-validation AUC: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}")
        return results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample data for testing the tabular model.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample data with all required columns
    """
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate sample data
    data = {
        'date': dates,
        'case_count': np.random.poisson(5, n_samples),
        'water_quality': np.random.normal(7.0, 0.5, n_samples),
        'rainfall': np.random.exponential(2, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'population_density': np.random.normal(500, 200, n_samples),
        'sanitation_index': np.random.uniform(0, 100, n_samples),
        'age_median': np.random.normal(30, 10, n_samples),
        'latitude': np.random.uniform(20, 30, n_samples),
        'longitude': np.random.uniform(70, 90, n_samples),
        'altitude': np.random.uniform(0, 1000, n_samples),
        'festival_flag': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'disaster_flag': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'outbreak_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def main():
    """Example usage of the tabular model."""
    # Create sample data
    data = create_sample_data(1000)
    
    # Initialize model
    model = TabularOutbreakPredictor()
    
    # Train model
    results = model.train(data)
    print("Training results:", results)
    
    # Test prediction
    sample_features = TabularFeatures(
        cases_7d_avg=5.0,
        cases_14d_avg=4.5,
        cases_30d_avg=4.0,
        case_trend=0.1,
        water_quality_7d_avg=7.0,
        rainfall_7d_total=10.0,
        temperature_7d_avg=25.0,
        population_density=500.0,
        sanitation_index=75.0,
        age_median=30.0,
        latitude=25.0,
        longitude=80.0,
        altitude=100.0,
        festival_flag=False,
        disaster_flag=False
    )
    
    prediction = model.predict(sample_features)
    print(f"Sample prediction: {prediction}")

if __name__ == "__main__":
    main()
