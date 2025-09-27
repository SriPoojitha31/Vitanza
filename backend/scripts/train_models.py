#!/usr/bin/env python3
"""
Model Training Script
====================

This script trains all models for the outbreak prediction system:
- Tabular model (XGBoost)
- Time-series model (LSTM)
- Text model (XLM-Roberta)
- Ensemble meta-learner

Usage:
    python train_models.py --data_path /path/to/data --output_dir /path/to/models
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.tabular_model import TabularOutbreakPredictor
from ml.timeseries_model import TimeSeriesOutbreakPredictor
from ml.text_model import MultilingualTextClassifier
from ml.ensemble_model import EnsembleOutbreakPredictor
from ml.preprocessing import DataPreprocessor
from schemas.ml_models import LanguageCode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training pipeline for outbreak prediction.
    
    This class orchestrates the training of all models in the ensemble:
    - Individual model training
    - Cross-validation and evaluation
    - Ensemble meta-learner training
    - Model persistence and metadata
    """
    
    def __init__(self, output_dir: str = "models", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save trained models
        config : Optional[Dict[str, Any]]
            Training configuration
        """
        self.output_dir = output_dir
        self.config = config or self._default_config()
        self.preprocessor = DataPreprocessor()
        self.training_results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration."""
        return {
            "tabular_model": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "early_stopping_rounds": 20,
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6
            },
            "timeseries_model": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 10,
                "sequence_length": 30,
                "input_dim": 10
            },
            "text_model": {
                "test_size": 0.2,
                "num_epochs": 3,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "max_length": 128
            },
            "ensemble_model": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "meta_learner_type": "logistic_regression"
            }
        }
    
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load training data from various sources.
        
        Parameters:
        -----------
        data_path : str
            Path to data directory or file
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Loaded datasets
        """
        logger.info(f"Loading data from {data_path}")
        
        datasets = {}
        
        # Load clinical data
        clinical_file = os.path.join(data_path, "clinical_data.csv")
        if os.path.exists(clinical_file):
            datasets["clinical"] = pd.read_csv(clinical_file)
            logger.info(f"Loaded clinical data: {datasets['clinical'].shape}")
        
        # Load environmental data
        environmental_file = os.path.join(data_path, "environmental_data.csv")
        if os.path.exists(environmental_file):
            datasets["environmental"] = pd.read_csv(environmental_file)
            logger.info(f"Loaded environmental data: {datasets['environmental'].shape}")
        
        # Load contextual data
        contextual_file = os.path.join(data_path, "contextual_data.csv")
        if os.path.exists(contextual_file):
            datasets["contextual"] = pd.read_csv(contextual_file)
            logger.info(f"Loaded contextual data: {datasets['contextual'].shape}")
        
        # Load combined dataset
        combined_file = os.path.join(data_path, "combined_data.csv")
        if os.path.exists(combined_file):
            datasets["combined"] = pd.read_csv(combined_file)
            logger.info(f"Loaded combined data: {datasets['combined'].shape}")
        
        # If no data files found, create sample data
        if not datasets:
            logger.warning("No data files found, creating sample data")
            datasets = self._create_sample_data()
        
        return datasets
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample data for training."""
        logger.info("Creating sample training data...")
        
        # Create sample clinical data
        n_samples = 1000
        np.random.seed(42)
        
        clinical_data = {
            'patient_age': np.random.randint(0, 100, n_samples),
            'patient_sex': np.random.choice(['male', 'female'], n_samples),
            'symptoms_text': [
                'Patient has fever and diarrhea' if i % 3 == 0 else 
                'রোগীর জ্বর এবং বমি' if i % 3 == 1 else
                'रोगी को बुखार और दस्त है'
                for i in range(n_samples)
            ],
            'symptoms_structured': [
                ['fever', 'diarrhea'] if i % 3 == 0 else
                ['fever', 'vomiting'] if i % 3 == 1 else
                ['fever', 'diarrhea']
                for i in range(n_samples)
            ],
            'severity': np.random.choice(['low', 'medium', 'high'], n_samples),
            'language': np.random.choice(['en', 'bn', 'hi'], n_samples),
            'facility_id': [f'facility_{i%10}' for i in range(n_samples)],
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'outbreak_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        clinical_df = pd.DataFrame(clinical_data)
        
        # Create sample environmental data
        environmental_data = {
            'ph': np.random.normal(7.0, 0.5, n_samples),
            'turbidity': np.random.exponential(1, n_samples),
            'temperature': np.random.normal(25, 5, n_samples),
            'conductivity': np.random.normal(500, 100, n_samples),
            'bacterial_test_result': np.random.choice([True, False], n_samples),
            'chlorine_residual': np.random.uniform(0, 2, n_samples),
            'quality_score': np.random.uniform(0, 100, n_samples),
            'location': [f'location_{i%20}' for i in range(n_samples)],
            'sensor_id': [f'sensor_{i%50}' for i in range(n_samples)],
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        }
        
        environmental_df = pd.DataFrame(environmental_data)
        
        # Create sample contextual data
        contextual_data = {
            'rainfall_24h': np.random.exponential(2, n_samples),
            'rainfall_7d': np.random.exponential(10, n_samples),
            'season': np.random.choice(['monsoon', 'summer', 'winter'], n_samples),
            'temperature_avg': np.random.normal(25, 5, n_samples),
            'humidity': np.random.uniform(40, 90, n_samples),
            'air_quality_index': np.random.uniform(0, 100, n_samples),
            'flood_risk': np.random.uniform(0, 1, n_samples),
            'drought_index': np.random.uniform(0, 1, n_samples),
            'population_count': np.random.randint(100, 10000, n_samples),
            'area_km2': np.random.uniform(1, 100, n_samples),
            'density_per_km2': np.random.uniform(10, 1000, n_samples),
            'toilet_coverage': np.random.uniform(0, 100, n_samples),
            'waste_management': np.random.uniform(0, 100, n_samples),
            'water_access': np.random.uniform(0, 100, n_samples),
            'hygiene_practices': np.random.uniform(0, 100, n_samples),
            'overall_sanitation_index': np.random.uniform(0, 100, n_samples),
            'festival_period': np.random.choice([True, False], n_samples),
            'natural_disaster': np.random.choice([True, False], n_samples),
            'disease_outbreak': np.random.choice([True, False], n_samples),
            'water_contamination': np.random.choice([True, False], n_samples),
            'health_campaign': np.random.choice([True, False], n_samples),
            'emergency_response': np.random.choice([True, False], n_samples),
            'village_ward': [f'village_{i%50}' for i in range(n_samples)],
            'district': [f'district_{i%10}' for i in range(n_samples)],
            'state': [f'state_{i%5}' for i in range(n_samples)],
            'latitude': np.random.uniform(20, 30, n_samples),
            'longitude': np.random.uniform(70, 90, n_samples),
            'altitude': np.random.uniform(0, 1000, n_samples)
        }
        
        contextual_df = pd.DataFrame(contextual_data)
        
        return {
            "clinical": clinical_df,
            "environmental": environmental_df,
            "contextual": contextual_df
        }
    
    def train_tabular_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the tabular XGBoost model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        logger.info("Training tabular model...")
        
        # Initialize model
        model_path = os.path.join(self.output_dir, "xgb_outbreak.json")
        model = TabularOutbreakPredictor(model_path)
        
        # Train model
        config = self.config["tabular_model"]
        results = model.train(
            data,
            target_column="outbreak_flag",
            test_size=config["test_size"],
            validation_size=config["validation_size"],
            early_stopping_rounds=config["early_stopping_rounds"],
            n_estimators=config["n_estimators"]
        )
        
        self.training_results["tabular"] = results
        logger.info(f"Tabular model training completed. AUC: {results['auc_roc']:.4f}")
        
        return results
    
    def train_timeseries_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the time-series LSTM model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        logger.info("Training time-series model...")
        
        # Initialize model
        model_path = os.path.join(self.output_dir, "ts_lstm.pth")
        config = self.config["timeseries_model"]
        
        model = TimeSeriesOutbreakPredictor(
            model_path=model_path,
            sequence_length=config["sequence_length"],
            input_dim=config["input_dim"]
        )
        
        # Train model
        results = model.train(
            data,
            target_column="outbreak_flag",
            test_size=config["test_size"],
            validation_size=config["validation_size"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            patience=config["patience"]
        )
        
        self.training_results["timeseries"] = results
        logger.info(f"Time-series model training completed. AUC: {results['test_auc']:.4f}")
        
        return results
    
    def train_text_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the multilingual text model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        logger.info("Training text model...")
        
        # Initialize model
        model_path = os.path.join(self.output_dir, "text_xlm_roberta")
        model = MultilingualTextClassifier(model_path=model_path)
        
        # Train model
        config = self.config["text_model"]
        results = model.train(
            data,
            text_column="symptoms_text",
            label_column="outbreak_flag",
            language_column="language",
            test_size=config["test_size"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"]
        )
        
        self.training_results["text"] = results
        logger.info(f"Text model training completed. AUC: {results['test_auc']:.4f}")
        
        return results
    
    def train_ensemble_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ensemble meta-learner.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        logger.info("Training ensemble model...")
        
        # Initialize ensemble model
        model_path = os.path.join(self.output_dir, "ensemble_meta.joblib")
        ensemble = EnsembleOutbreakPredictor(model_path)
        
        # Load base models
        tabular_model = TabularOutbreakPredictor(os.path.join(self.output_dir, "xgb_outbreak.json"))
        tabular_model.load_model()
        
        timeseries_model = TimeSeriesOutbreakPredictor(os.path.join(self.output_dir, "ts_lstm.pth"))
        timeseries_model.load_model()
        
        text_model = MultilingualTextClassifier(os.path.join(self.output_dir, "text_xlm_roberta"))
        text_model.load_model()
        
        # Set base models
        ensemble.set_base_models(tabular_model, timeseries_model, text_model)
        
        # Train meta-learner
        config = self.config["ensemble_model"]
        results = ensemble.train_meta_learner(
            data,
            target_column="outbreak_flag",
            test_size=config["test_size"],
            validation_size=config["validation_size"],
            meta_learner_type=config["meta_learner_type"]
        )
        
        self.training_results["ensemble"] = results
        logger.info(f"Ensemble model training completed. AUC: {results['test_auc']:.4f}")
        
        return results
    
    def train_all_models(self, data_path: str) -> Dict[str, Any]:
        """
        Train all models in the pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to training data
            
        Returns:
        --------
        Dict[str, Any]
            Training results for all models
        """
        logger.info("Starting comprehensive model training pipeline...")
        
        # Load data
        datasets = self.load_data(data_path)
        
        # Prepare combined dataset
        if "combined" in datasets:
            combined_data = datasets["combined"]
        else:
            # Combine individual datasets
            combined_data = self._combine_datasets(datasets)
        
        # Train individual models
        logger.info("Training individual models...")
        
        # Tabular model
        tabular_results = self.train_tabular_model(combined_data)
        
        # Time-series model
        timeseries_results = self.train_timeseries_model(combined_data)
        
        # Text model
        text_results = self.train_text_model(combined_data)
        
        # Ensemble model
        ensemble_results = self.train_ensemble_model(combined_data)
        
        # Save training metadata
        self._save_training_metadata()
        
        logger.info("All models trained successfully!")
        return self.training_results
    
    def _combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into a single training dataset."""
        logger.info("Combining datasets...")
        
        # Start with clinical data as base
        if "clinical" in datasets:
            combined = datasets["clinical"].copy()
        else:
            # Create empty DataFrame if no clinical data
            combined = pd.DataFrame()
        
        # Add environmental features
        if "environmental" in datasets:
            env_data = datasets["environmental"]
            # Aggregate environmental data by time
            env_agg = env_data.groupby('timestamp').agg({
                'ph': 'mean',
                'turbidity': 'mean',
                'temperature': 'mean',
                'conductivity': 'mean',
                'quality_score': 'mean'
            }).reset_index()
            
            if not combined.empty:
                combined = combined.merge(env_agg, left_on='date', right_on='timestamp', how='left')
            else:
                combined = env_agg
        
        # Add contextual features
        if "contextual" in datasets:
            context_data = datasets["contextual"]
            # Take mean of contextual features
            context_agg = context_data.mean().to_frame().T
            
            for col in context_agg.columns:
                combined[col] = context_agg[col].iloc[0]
        
        # Ensure we have target column
        if "outbreak_flag" not in combined.columns:
            combined["outbreak_flag"] = np.random.choice([0, 1], len(combined), p=[0.7, 0.3])
        
        logger.info(f"Combined dataset shape: {combined.shape}")
        return combined
    
    def _save_training_metadata(self):
        """Save training metadata and results."""
        metadata = {
            "training_timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": self.training_results,
            "model_paths": {
                "tabular": os.path.join(self.output_dir, "xgb_outbreak.json"),
                "timeseries": os.path.join(self.output_dir, "ts_lstm.pth"),
                "text": os.path.join(self.output_dir, "text_xlm_roberta"),
                "ensemble": os.path.join(self.output_dir, "ensemble_meta.joblib")
            }
        }
        
        metadata_path = os.path.join(self.output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data.
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test dataset
            
        Returns:
        --------
        Dict[str, Any]
            Evaluation results
        """
        logger.info("Evaluating models on test data...")
        
        evaluation_results = {}
        
        # Load and evaluate each model
        try:
            # Tabular model evaluation
            tabular_model = TabularOutbreakPredictor(os.path.join(self.output_dir, "xgb_outbreak.json"))
            tabular_model.load_model()
            
            # Create tabular features
            tabular_features = self.preprocessor.create_tabular_features(
                test_data, test_data, test_data
            )
            
            # Make predictions
            tabular_predictions = tabular_model.predict_batch(
                [tabular_features.iloc[i].to_dict() for i in range(len(tabular_features))]
            )
            
            evaluation_results["tabular"] = {
                "predictions": tabular_predictions,
                "mean_prediction": np.mean(tabular_predictions),
                "std_prediction": np.std(tabular_predictions)
            }
            
        except Exception as e:
            logger.error(f"Tabular model evaluation failed: {e}")
            evaluation_results["tabular"] = {"error": str(e)}
        
        # Similar evaluation for other models...
        # (Implementation would continue for timeseries, text, and ensemble models)
        
        return evaluation_results

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train outbreak prediction models")
    parser.add_argument("--data_path", type=str, default="data", 
                       help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Directory to save trained models")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to training configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(output_dir=args.output_dir, config=config)
    
    # Train all models
    try:
        results = trainer.train_all_models(args.data_path)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        for model_name, result in results.items():
            if "auc" in result:
                print(f"{model_name.upper()}: AUC = {result['auc']:.4f}")
            elif "test_auc" in result:
                print(f"{model_name.upper()}: AUC = {result['test_auc']:.4f}")
            else:
                print(f"{model_name.upper()}: Training completed")
        
        print(f"\nModels saved to: {args.output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
