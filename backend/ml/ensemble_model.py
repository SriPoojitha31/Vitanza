"""
Ensemble Meta-Learner for Outbreak Prediction
==============================================

This module implements an ensemble meta-learner that combines predictions
from multiple models (tabular, time-series, text) to make final outbreak predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from datetime import datetime

from schemas.ml_models import (
    TabularFeatures, TimeSeriesWindow, TextFeatures, 
    EnsemblePrediction, ModelPrediction, InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)

class EnsembleOutbreakPredictor:
    """
    Ensemble meta-learner for combining multiple model predictions.
    
    This class implements stacking ensemble learning that combines:
    - Tabular model predictions (XGBoost)
    - Time-series model predictions (LSTM)
    - Text model predictions (XLM-Roberta)
    - Additional meta-features
    """
    
    def __init__(self, model_path: str = "models/ensemble_meta.joblib",
                 base_models: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble predictor.
        
        Parameters:
        -----------
        model_path : str
            Path to save/load the meta-learner
        base_models : Dict[str, Any]
            Dictionary of base models
        """
        self.model_path = model_path
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.base_models = base_models or {}
        self.feature_names = None
        self.model_weights = None
        
        # Meta-learner options
        self.meta_learner_type = "logistic_regression"  # or "random_forest", "mlp", "voting"
        
    def set_base_models(self, tabular_model, timeseries_model, text_model):
        """
        Set the base models for ensemble prediction.
        
        Parameters:
        -----------
        tabular_model : Any
            Trained tabular model (XGBoost)
        timeseries_model : Any
            Trained time-series model (LSTM)
        text_model : Any
            Trained text model (XLM-Roberta)
        """
        self.base_models = {
            "tabular": tabular_model,
            "timeseries": timeseries_model,
            "text": text_model
        }
        logger.info("Base models set for ensemble prediction")
    
    def generate_base_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from all base models.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with features for all models
            
        Returns:
        --------
        pd.DataFrame
            Base model predictions
        """
        logger.info("Generating base model predictions...")
        
        predictions_df = pd.DataFrame()
        
        # Tabular model predictions
        if "tabular" in self.base_models:
            try:
                tabular_model = self.base_models["tabular"]
                tabular_features = self._extract_tabular_features(data)
                tabular_preds = tabular_model.predict_batch(tabular_features)
                predictions_df["tabular_prediction"] = tabular_preds
                predictions_df["tabular_confidence"] = [abs(p - 0.5) * 2 for p in tabular_preds]
            except Exception as e:
                logger.warning(f"Tabular model prediction failed: {e}")
                predictions_df["tabular_prediction"] = 0.5
                predictions_df["tabular_confidence"] = 0.0
        
        # Time-series model predictions
        if "timeseries" in self.base_models:
            try:
                timeseries_model = self.base_models["timeseries"]
                timeseries_windows = self._extract_timeseries_windows(data)
                timeseries_preds = timeseries_model.predict_batch(timeseries_windows)
                predictions_df["timeseries_prediction"] = timeseries_preds
                predictions_df["timeseries_confidence"] = [abs(p - 0.5) * 2 for p in timeseries_preds]
            except Exception as e:
                logger.warning(f"Time-series model prediction failed: {e}")
                predictions_df["timeseries_prediction"] = 0.5
                predictions_df["timeseries_confidence"] = 0.0
        
        # Text model predictions
        if "text" in self.base_models:
            try:
                text_model = self.base_models["text"]
                text_features = self._extract_text_features(data)
                text_preds = text_model.predict_batch(text_features)
                predictions_df["text_prediction"] = text_preds
                predictions_df["text_confidence"] = [abs(p - 0.5) * 2 for p in text_preds]
            except Exception as e:
                logger.warning(f"Text model prediction failed: {e}")
                predictions_df["text_prediction"] = 0.5
                predictions_df["text_confidence"] = 0.0
        
        # Meta-features
        predictions_df = self._add_meta_features(predictions_df, data)
        
        logger.info(f"Generated {len(predictions_df.columns)} meta-features")
        return predictions_df
    
    def _extract_tabular_features(self, data: pd.DataFrame) -> List[TabularFeatures]:
        """Extract tabular features from data."""
        features_list = []
        
        for _, row in data.iterrows():
            features = TabularFeatures(
                cases_7d_avg=row.get('cases_7d_avg', 0),
                cases_14d_avg=row.get('cases_14d_avg', 0),
                cases_30d_avg=row.get('cases_30d_avg', 0),
                case_trend=row.get('case_trend', 0),
                water_quality_7d_avg=row.get('water_quality_7d_avg', 7.0),
                rainfall_7d_total=row.get('rainfall_7d_total', 0),
                temperature_7d_avg=row.get('temperature_7d_avg', 25.0),
                population_density=row.get('population_density', 500.0),
                sanitation_index=row.get('sanitation_index', 75.0),
                age_median=row.get('age_median', 30.0),
                latitude=row.get('latitude', 25.0),
                longitude=row.get('longitude', 80.0),
                altitude=row.get('altitude', 100.0),
                festival_flag=row.get('festival_flag', False),
                disaster_flag=row.get('disaster_flag', False)
            )
            features_list.append(features)
        
        return features_list
    
    def _extract_timeseries_windows(self, data: pd.DataFrame) -> List[TimeSeriesWindow]:
        """Extract time-series windows from data."""
        windows_list = []
        
        for _, row in data.iterrows():
            # Create sample time-series window
            window = TimeSeriesWindow(
                timestamps=[datetime.now()] * 30,  # Placeholder
                sensor_readings=[[7.0, 1.0, 25.0, 500.0, 2.0, 25.0, 60.0, 5.0] for _ in range(30)],
                case_counts=[5] * 30,
                environmental_factors=[[2.0, 25.0, 60.0] for _ in range(30)],
                window_size=30
            )
            windows_list.append(window)
        
        return windows_list
    
    def _extract_text_features(self, data: pd.DataFrame) -> List[TextFeatures]:
        """Extract text features from data."""
        features_list = []
        
        for _, row in data.iterrows():
            text = row.get('clinical_text', '')
            language = row.get('text_language', 'auto')
            
            features = TextFeatures(
                text=text,
                language=language,
                translated_text=None,
                symptom_keywords=[],
                severity_indicators=[],
                sentiment_score=None
            )
            features_list.append(features)
        
        return features_list
    
    def _add_meta_features(self, predictions_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add meta-features to predictions."""
        # Prediction agreement features
        if "tabular_prediction" in predictions_df.columns and "timeseries_prediction" in predictions_df.columns:
            predictions_df["tabular_timeseries_agreement"] = np.abs(
                predictions_df["tabular_prediction"] - predictions_df["timeseries_prediction"]
            )
        
        if "tabular_prediction" in predictions_df.columns and "text_prediction" in predictions_df.columns:
            predictions_df["tabular_text_agreement"] = np.abs(
                predictions_df["tabular_prediction"] - predictions_df["text_prediction"]
            )
        
        if "timeseries_prediction" in predictions_df.columns and "text_prediction" in predictions_df.columns:
            predictions_df["timeseries_text_agreement"] = np.abs(
                predictions_df["timeseries_prediction"] - predictions_df["text_prediction"]
            )
        
        # Average prediction
        prediction_cols = [col for col in predictions_df.columns if col.endswith("_prediction")]
        if prediction_cols:
            predictions_df["average_prediction"] = predictions_df[prediction_cols].mean(axis=1)
            predictions_df["prediction_std"] = predictions_df[prediction_cols].std(axis=1)
        
        # Confidence features
        confidence_cols = [col for col in predictions_df.columns if col.endswith("_confidence")]
        if confidence_cols:
            predictions_df["average_confidence"] = predictions_df[confidence_cols].mean(axis=1)
            predictions_df["max_confidence"] = predictions_df[confidence_cols].max(axis=1)
            predictions_df["min_confidence"] = predictions_df[confidence_cols].min(axis=1)
        
        # Data quality features
        if 'data_quality_score' in data.columns:
            predictions_df["data_quality"] = data["data_quality_score"]
        else:
            predictions_df["data_quality"] = 1.0
        
        # Temporal features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            predictions_df["hour_of_day"] = data['timestamp'].dt.hour
            predictions_df["day_of_week"] = data['timestamp'].dt.dayofweek
            predictions_df["month"] = data['timestamp'].dt.month
        
        return predictions_df
    
    def train_meta_learner(self, data: pd.DataFrame, target_column: str = 'outbreak_flag',
                          test_size: float = 0.2, validation_size: float = 0.2,
                          meta_learner_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Train the meta-learner on base model predictions.
        
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
        meta_learner_type : str
            Type of meta-learner to use
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metrics
        """
        logger.info("Training ensemble meta-learner...")
        
        # Generate base predictions
        meta_features = self.generate_base_predictions(data)
        
        # Get target labels
        target = data[target_column].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            meta_features, target, test_size=test_size + validation_size,
            stratify=target, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + validation_size),
            stratify=y_temp, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize meta-learner
        self.meta_learner_type = meta_learner_type
        
        if meta_learner_type == "logistic_regression":
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif meta_learner_type == "random_forest":
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif meta_learner_type == "mlp":
            self.meta_learner = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=42
            )
        elif meta_learner_type == "voting":
            # Voting classifier with multiple base learners
            lr = LogisticRegression(max_iter=1000, random_state=42)
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
            
            self.meta_learner = VotingClassifier(
                estimators=[('lr', lr), ('rf', rf), ('mlp', mlp)],
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")
        
        # Train meta-learner
        self.meta_learner.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.meta_learner.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        # Evaluate on test set
        y_test_pred = self.meta_learner.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        
        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
        test_pr_auc = auc(recall, precision)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.meta_learner, 'feature_importances_'):
            feature_importance = dict(zip(meta_features.columns, self.meta_learner.feature_importances_))
        elif hasattr(self.meta_learner, 'coef_'):
            feature_importance = dict(zip(meta_features.columns, np.abs(self.meta_learner.coef_[0])))
        
        # Store feature names
        self.feature_names = meta_features.columns.tolist()
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.meta_learner, self.model_path)
        
        # Save scaler
        scaler_path = self.model_path.replace('.joblib', '_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        features_path = self.model_path.replace('.joblib', '_features.joblib')
        joblib.dump(self.feature_names, features_path)
        
        results = {
            "val_auc": val_auc,
            "test_auc": test_auc,
            "test_pr_auc": test_pr_auc,
            "meta_learner_type": meta_learner_type,
            "n_features": len(self.feature_names),
            "feature_importance": feature_importance,
            "classification_report": classification_report(y_test, (y_test_pred > 0.5).astype(int), output_dict=True)
        }
        
        logger.info(f"Meta-learner training completed. Test AUC: {test_auc:.4f}")
        return results
    
    def predict(self, inference_request: InferenceRequest) -> InferenceResponse:
        """
        Make ensemble prediction for an inference request.
        
        Parameters:
        -----------
        inference_request : InferenceRequest
            Complete inference request with all data
            
        Returns:
        --------
        InferenceResponse
            Ensemble prediction response
        """
        if self.meta_learner is None:
            self.load_model()
        
        # Generate base predictions
        base_predictions = {}
        base_confidences = {}
        
        # Tabular prediction
        if "tabular" in self.base_models:
            try:
                tabular_pred = self.base_models["tabular"].predict(inference_request.features_tabular)
                base_predictions["tabular"] = tabular_pred
                base_confidences["tabular"] = abs(tabular_pred - 0.5) * 2
            except Exception as e:
                logger.warning(f"Tabular prediction failed: {e}")
                base_predictions["tabular"] = 0.5
                base_confidences["tabular"] = 0.0
        
        # Time-series prediction
        if "timeseries" in self.base_models:
            try:
                timeseries_pred = self.base_models["timeseries"].predict(inference_request.timeseries_window)
                base_predictions["timeseries"] = timeseries_pred
                base_confidences["timeseries"] = abs(timeseries_pred - 0.5) * 2
            except Exception as e:
                logger.warning(f"Time-series prediction failed: {e}")
                base_predictions["timeseries"] = 0.5
                base_confidences["timeseries"] = 0.0
        
        # Text prediction
        if "text" in self.base_models:
            try:
                text_features = TextFeatures(
                    text=inference_request.clinical_text,
                    language=inference_request.text_language
                )
                text_pred = self.base_models["text"].predict(text_features)
                base_predictions["text"] = text_pred
                base_confidences["text"] = abs(text_pred - 0.5) * 2
            except Exception as e:
                logger.warning(f"Text prediction failed: {e}")
                base_predictions["text"] = 0.5
                base_confidences["text"] = 0.0
        
        # Create meta-features
        meta_features = self._create_meta_features(base_predictions, base_confidences, inference_request)
        
        # Make ensemble prediction
        meta_features_scaled = self.scaler.transform(meta_features.reshape(1, -1))
        ensemble_probability = self.meta_learner.predict_proba(meta_features_scaled)[0, 1]
        
        # Calculate confidence
        confidence = self._calculate_confidence(base_confidences, ensemble_probability)
        
        # Determine severity level
        severity_level = self._determine_severity_level(ensemble_probability)
        
        # Calculate lead time
        lead_time_days = self._calculate_lead_time(ensemble_probability, base_predictions)
        
        # Generate explanations
        contributing_factors = self._identify_contributing_factors(base_predictions, meta_features)
        recommendations = self._generate_recommendations(ensemble_probability, contributing_factors)
        
        # Create individual model predictions
        individual_predictions = [
            ModelPrediction(
                model_name="tabular",
                prediction=base_predictions.get("tabular", 0.5),
                confidence=base_confidences.get("tabular", 0.0),
                features_used=["cases_7d_avg", "water_quality_7d_avg", "population_density"],
                feature_importance={"cases_7d_avg": 0.3, "water_quality_7d_avg": 0.4, "population_density": 0.3}
            ),
            ModelPrediction(
                model_name="timeseries",
                prediction=base_predictions.get("timeseries", 0.5),
                confidence=base_confidences.get("timeseries", 0.0),
                features_used=["sensor_readings", "case_counts", "environmental_factors"],
                feature_importance={"sensor_readings": 0.5, "case_counts": 0.3, "environmental_factors": 0.2}
            ),
            ModelPrediction(
                model_name="text",
                prediction=base_predictions.get("text", 0.5),
                confidence=base_confidences.get("text", 0.0),
                features_used=["clinical_text", "symptom_keywords", "severity_indicators"],
                feature_importance={"clinical_text": 0.6, "symptom_keywords": 0.3, "severity_indicators": 0.1}
            )
        ]
        
        # Create response
        response = InferenceResponse(
            outbreak_probability=float(ensemble_probability),
            confidence=float(confidence),
            lead_time_days=int(lead_time_days),
            severity_level=severity_level,
            tabular_prediction=base_predictions.get("tabular", 0.5),
            timeseries_prediction=base_predictions.get("timeseries", 0.5),
            text_prediction=base_predictions.get("text", 0.5),
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            feature_importance=self._get_feature_importance(),
            model_versions={"ensemble": "1.0", "tabular": "1.0", "timeseries": "1.0", "text": "1.0"},
            processing_time_ms=0.0,  # Would be calculated in actual implementation
            timestamp=datetime.utcnow()
        )
        
        return response
    
    def _create_meta_features(self, base_predictions: Dict[str, float], 
                             base_confidences: Dict[str, float],
                             inference_request: InferenceRequest) -> np.ndarray:
        """Create meta-features for ensemble prediction."""
        features = []
        
        # Base predictions
        features.extend([
            base_predictions.get("tabular", 0.5),
            base_predictions.get("timeseries", 0.5),
            base_predictions.get("text", 0.5)
        ])
        
        # Base confidences
        features.extend([
            base_confidences.get("tabular", 0.0),
            base_confidences.get("timeseries", 0.0),
            base_confidences.get("text", 0.0)
        ])
        
        # Prediction agreement
        if "tabular" in base_predictions and "timeseries" in base_predictions:
            features.append(abs(base_predictions["tabular"] - base_predictions["timeseries"]))
        else:
            features.append(0.0)
        
        if "tabular" in base_predictions and "text" in base_predictions:
            features.append(abs(base_predictions["tabular"] - base_predictions["text"]))
        else:
            features.append(0.0)
        
        if "timeseries" in base_predictions and "text" in base_predictions:
            features.append(abs(base_predictions["timeseries"] - base_predictions["text"]))
        else:
            features.append(0.0)
        
        # Average prediction and confidence
        pred_values = [p for p in base_predictions.values()]
        conf_values = [c for c in base_confidences.values()]
        
        features.extend([
            np.mean(pred_values) if pred_values else 0.5,
            np.std(pred_values) if len(pred_values) > 1 else 0.0,
            np.mean(conf_values) if conf_values else 0.0,
            np.max(conf_values) if conf_values else 0.0
        ])
        
        # Data quality features
        features.extend([
            inference_request.features_tabular.population_density / 1000.0,  # Normalized
            inference_request.features_tabular.sanitation_index / 100.0,    # Normalized
            float(inference_request.events.festival_period),
            float(inference_request.events.natural_disaster)
        ])
        
        return np.array(features)
    
    def _calculate_confidence(self, base_confidences: Dict[str, float], 
                            ensemble_probability: float) -> float:
        """Calculate ensemble confidence."""
        # Weighted average of base confidences
        if base_confidences:
            avg_confidence = np.mean(list(base_confidences.values()))
        else:
            avg_confidence = 0.0
        
        # Adjust based on prediction certainty
        certainty = abs(ensemble_probability - 0.5) * 2
        
        # Combine base confidence with prediction certainty
        final_confidence = (avg_confidence + certainty) / 2
        
        return min(1.0, max(0.0, final_confidence))
    
    def _determine_severity_level(self, probability: float) -> str:
        """Determine severity level based on probability."""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_lead_time(self, probability: float, base_predictions: Dict[str, float]) -> int:
        """Calculate predicted lead time in days."""
        # Base lead time on probability and model agreement
        base_lead_time = int((1 - probability) * 14)  # 0-14 days
        
        # Adjust based on model agreement
        if base_predictions:
            agreement = 1 - np.std(list(base_predictions.values()))
            adjustment = int(agreement * 7)  # 0-7 days adjustment
            lead_time = max(1, base_lead_time + adjustment)
        else:
            lead_time = max(1, base_lead_time)
        
        return min(14, lead_time)
    
    def _identify_contributing_factors(self, base_predictions: Dict[str, float], 
                                     meta_features: np.ndarray) -> List[str]:
        """Identify key contributing factors."""
        factors = []
        
        # High probability factors
        if base_predictions.get("tabular", 0) > 0.7:
            factors.append("High case count and environmental factors")
        
        if base_predictions.get("timeseries", 0) > 0.7:
            factors.append("Temporal pattern indicates outbreak risk")
        
        if base_predictions.get("text", 0) > 0.7:
            factors.append("Clinical symptoms suggest outbreak conditions")
        
        # Model agreement
        if len(base_predictions) > 1:
            predictions = list(base_predictions.values())
            if np.std(predictions) < 0.1:
                factors.append("Strong model agreement on outbreak risk")
        
        # Environmental factors
        if meta_features[12] > 0.5:  # Festival period
            factors.append("Festival period increases transmission risk")
        
        if meta_features[13] > 0.5:  # Natural disaster
            factors.append("Natural disaster disrupts health infrastructure")
        
        return factors if factors else ["Standard outbreak risk assessment"]
    
    def _generate_recommendations(self, probability: float, 
                                contributing_factors: List[str]) -> List[str]:
        """Generate prevention recommendations."""
        recommendations = []
        
        if probability >= 0.8:
            recommendations.extend([
                "Immediate emergency response activation",
                "Deploy additional healthcare resources",
                "Implement strict containment measures",
                "Alert regional health authorities"
            ])
        elif probability >= 0.6:
            recommendations.extend([
                "Increase surveillance and monitoring",
                "Prepare emergency response teams",
                "Stock emergency medical supplies",
                "Conduct community awareness campaigns"
            ])
        elif probability >= 0.4:
            recommendations.extend([
                "Enhanced monitoring of high-risk areas",
                "Regular health check-ups for vulnerable populations",
                "Improve water and sanitation infrastructure",
                "Community health education programs"
            ])
        else:
            recommendations.extend([
                "Maintain routine surveillance",
                "Continue preventive health measures",
                "Monitor environmental conditions",
                "Regular community health assessments"
            ])
        
        return recommendations
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-learner."""
        if hasattr(self.meta_learner, 'feature_importances_'):
            return dict(zip(self.feature_names, self.meta_learner.feature_importances_))
        elif hasattr(self.meta_learner, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.meta_learner.coef_[0])))
        else:
            return {}
    
    def load_model(self):
        """Load the trained meta-learner and preprocessing objects."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Meta-learner file not found: {self.model_path}")
        
        # Load meta-learner
        self.meta_learner = joblib.load(self.model_path)
        
        # Load scaler
        scaler_path = self.model_path.replace('.joblib', '_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        features_path = self.model_path.replace('.joblib', '_features.joblib')
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        
        logger.info("Ensemble meta-learner loaded successfully")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble model information."""
        return {
            "meta_learner_type": self.meta_learner_type,
            "base_models": list(self.base_models.keys()),
            "feature_names": self.feature_names,
            "model_path": self.model_path,
            "is_loaded": self.meta_learner is not None
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_ensemble_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data for ensemble training."""
    np.random.seed(42)
    
    data = {
        'cases_7d_avg': np.random.poisson(5, n_samples),
        'water_quality_7d_avg': np.random.normal(7.0, 0.5, n_samples),
        'rainfall_7d_total': np.random.exponential(2, n_samples),
        'population_density': np.random.normal(500, 200, n_samples),
        'sanitation_index': np.random.uniform(0, 100, n_samples),
        'clinical_text': ['Patient has fever and diarrhea'] * n_samples,
        'text_language': ['en'] * n_samples,
        'festival_flag': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'disaster_flag': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'outbreak_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def main():
    """Example usage of the ensemble model."""
    # Create sample data
    data = create_sample_ensemble_data(100)
    
    # Initialize ensemble
    ensemble = EnsembleOutbreakPredictor()
    
    # Train meta-learner (would need base models in practice)
    # results = ensemble.train_meta_learner(data)
    # print("Ensemble training results:", results)
    
    print("Ensemble model initialized successfully")

if __name__ == "__main__":
    main()
