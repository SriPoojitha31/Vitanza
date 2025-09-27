"""
Data Preprocessing and Feature Engineering Pipeline
==================================================

This module provides comprehensive data preprocessing and feature engineering
for the outbreak prediction system, handling multiple data modalities and
ensuring data quality and consistency.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
import re
import unicodedata

from schemas.ml_models import (
    ClinicalReport, WaterQualityReading, EnvironmentalContext,
    PopulationDensity, SanitationIndex, EventFlags, GeographicLocation,
    TabularFeatures, TimeSeriesWindow, TextFeatures, LanguageCode
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for outbreak prediction.
    
    This class handles preprocessing for multiple data modalities:
    - Clinical reports and symptoms
    - Environmental and sensor data
    - Population and demographic data
    - Geographic and contextual data
    - Text data and language processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data preprocessor.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Configuration parameters for preprocessing
        """
        self.config = config or self._default_config()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.is_fitted = False
        
    def _default_config(self) -> Dict[str, Any]:
        """Default preprocessing configuration."""
        return {
            "missing_value_strategy": "median",  # "mean", "median", "most_frequent", "knn"
            "outlier_detection": True,
            "outlier_method": "iqr",  # "iqr", "zscore", "isolation_forest"
            "outlier_threshold": 3.0,
            "feature_scaling": "standard",  # "standard", "minmax", "robust"
            "feature_selection": True,
            "feature_selection_k": 20,
            "text_preprocessing": True,
            "language_detection": True,
            "temporal_features": True,
            "interaction_features": True
        }
    
    def preprocess_clinical_data(self, reports: List[ClinicalReport]) -> pd.DataFrame:
        """
        Preprocess clinical reports data.
        
        Parameters:
        -----------
        reports : List[ClinicalReport]
            List of clinical reports
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed clinical data
        """
        logger.info(f"Preprocessing {len(reports)} clinical reports...")
        
        data = []
        for report in reports:
            # Basic features
            row = {
                "patient_age": report.age,
                "patient_sex": report.sex.value,
                "symptoms_text": report.symptoms_text,
                "symptoms_structured": report.symptoms_structured,
                "severity": report.severity.value if report.severity else "unknown",
                "language": report.language.value,
                "facility_id": report.facility_id,
                "date": report.date,
                "diagnosis": report.diagnosis,
                "treatment": report.treatment,
                "outcome": report.outcome
            }
            
            # Extract structured symptoms
            if report.symptoms_structured:
                row["symptom_count"] = len(report.symptoms_structured)
                row["has_fever"] = any("fever" in s.lower() for s in report.symptoms_structured)
                row["has_diarrhea"] = any("diarrhea" in s.lower() for s in report.symptoms_structured)
                row["has_vomiting"] = any("vomit" in s.lower() for s in report.symptoms_structured)
                row["has_respiratory"] = any(s in ["cough", "breathing", "respiratory"] for s in report.symptoms_structured)
            else:
                row["symptom_count"] = 0
                row["has_fever"] = False
                row["has_diarrhea"] = False
                row["has_vomiting"] = False
                row["has_respiratory"] = False
            
            # Age groups
            if report.age < 5:
                row["age_group"] = "infant"
            elif report.age < 18:
                row["age_group"] = "child"
            elif report.age < 65:
                row["age_group"] = "adult"
            else:
                row["age_group"] = "elderly"
            
            # Text features
            if self.config["text_preprocessing"]:
                text_features = self._extract_text_features(report.symptoms_text, report.language)
                row.update(text_features)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        logger.info(f"Preprocessed clinical data: {df.shape}")
        return df
    
    def preprocess_environmental_data(self, readings: List[WaterQualityReading]) -> pd.DataFrame:
        """
        Preprocess environmental and sensor data.
        
        Parameters:
        -----------
        readings : List[WaterQualityReading]
            List of environmental readings
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed environmental data
        """
        logger.info(f"Preprocessing {len(readings)} environmental readings...")
        
        data = []
        for reading in readings:
            row = {
                "ph": reading.ph,
                "turbidity": reading.turbidity,
                "temperature": reading.temperature,
                "conductivity": reading.conductivity,
                "bacterial_test": reading.bacterial_test_result,
                "chlorine_residual": reading.chlorine_residual,
                "quality_score": reading.quality_score,
                "location": reading.location,
                "sensor_id": reading.sensor_id,
                "timestamp": reading.timestamp
            }
            
            # Calculate derived features
            row["ph_category"] = self._categorize_ph(reading.ph)
            row["turbidity_category"] = self._categorize_turbidity(reading.turbidity)
            row["temperature_category"] = self._categorize_temperature(reading.temperature)
            
            # Water quality index
            row["water_quality_index"] = self._calculate_water_quality_index(reading)
            
            # Contamination risk
            row["contamination_risk"] = self._calculate_contamination_risk(reading)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Detect and handle outliers
        if self.config["outlier_detection"]:
            df = self._handle_outliers(df)
        
        logger.info(f"Preprocessed environmental data: {df.shape}")
        return df
    
    def preprocess_contextual_data(self, 
                                 environmental: List[EnvironmentalContext],
                                 population: List[PopulationDensity],
                                 sanitation: List[SanitationIndex],
                                 events: List[EventFlags],
                                 locations: List[GeographicLocation]) -> pd.DataFrame:
        """
        Preprocess contextual and demographic data.
        
        Parameters:
        -----------
        environmental : List[EnvironmentalContext]
            Environmental context data
        population : List[PopulationDensity]
            Population density data
        sanitation : List[SanitationIndex]
            Sanitation index data
        events : List[EventFlags]
            Event flags data
        locations : List[GeographicLocation]
            Geographic location data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed contextual data
        """
        logger.info("Preprocessing contextual data...")
        
        # Combine all contextual data
        data = []
        max_len = max(len(env) for env in [environmental, population, sanitation, events, locations])
        
        for i in range(max_len):
            row = {}
            
            # Environmental context
            if i < len(environmental):
                env = environmental[i]
                row.update({
                    "rainfall_24h": env.rainfall_24h,
                    "rainfall_7d": env.rainfall_7d,
                    "season": env.season,
                    "temperature_avg": env.temperature_avg,
                    "humidity": env.humidity,
                    "air_quality_index": env.air_quality_index,
                    "flood_risk": env.flood_risk,
                    "drought_index": env.drought_index
                })
            
            # Population data
            if i < len(population):
                pop = population[i]
                row.update({
                    "population_count": pop.population_count,
                    "area_km2": pop.area_km2,
                    "density_per_km2": pop.density_per_km2,
                    "age_distribution": pop.age_distribution
                })
            
            # Sanitation data
            if i < len(sanitation):
                san = sanitation[i]
                row.update({
                    "toilet_coverage": san.toilet_coverage,
                    "waste_management": san.waste_management,
                    "water_access": san.water_access,
                    "hygiene_practices": san.hygiene_practices,
                    "overall_sanitation_index": san.overall_index
                })
            
            # Event flags
            if i < len(events):
                event = events[i]
                row.update({
                    "festival_period": event.festival_period,
                    "natural_disaster": event.natural_disaster,
                    "disease_outbreak": event.disease_outbreak,
                    "water_contamination": event.water_contamination,
                    "health_campaign": event.health_campaign,
                    "emergency_response": event.emergency_response
                })
            
            # Geographic data
            if i < len(locations):
                loc = locations[i]
                row.update({
                    "village_ward": loc.village_ward,
                    "district": loc.district,
                    "state": loc.state,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "altitude": loc.altitude
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create derived features
        if self.config["temporal_features"]:
            df = self._create_temporal_features(df)
        
        if self.config["interaction_features"]:
            df = self._create_interaction_features(df)
        
        logger.info(f"Preprocessed contextual data: {df.shape}")
        return df
    
    def create_tabular_features(self, clinical_df: pd.DataFrame, 
                              environmental_df: pd.DataFrame,
                              contextual_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated tabular features for XGBoost model.
        
        Parameters:
        -----------
        clinical_df : pd.DataFrame
            Preprocessed clinical data
        environmental_df : pd.DataFrame
            Preprocessed environmental data
        contextual_df : pd.DataFrame
            Preprocessed contextual data
            
        Returns:
        --------
        pd.DataFrame
            Aggregated tabular features
        """
        logger.info("Creating tabular features...")
        
        features = {}
        
        # Clinical features (aggregated by time window)
        if not clinical_df.empty:
            clinical_df['date'] = pd.to_datetime(clinical_df['date'])
            
            # 7-day rolling features
            features['cases_7d_avg'] = clinical_df.groupby('date')['patient_age'].count().rolling(7).mean()
            features['cases_14d_avg'] = clinical_df.groupby('date')['patient_age'].count().rolling(14).mean()
            features['cases_30d_avg'] = clinical_df.groupby('date')['patient_age'].count().rolling(30).mean()
            
            # Case trend
            case_counts = clinical_df.groupby('date')['patient_age'].count()
            features['case_trend'] = case_counts.rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0)
            
            # Symptom features
            features['fever_rate_7d'] = clinical_df.groupby('date')['has_fever'].mean().rolling(7).mean()
            features['diarrhea_rate_7d'] = clinical_df.groupby('date')['has_diarrhea'].mean().rolling(7).mean()
            features['respiratory_rate_7d'] = clinical_df.groupby('date')['has_respiratory'].mean().rolling(7).mean()
        
        # Environmental features
        if not environmental_df.empty:
            environmental_df['timestamp'] = pd.to_datetime(environmental_df['timestamp'])
            
            features['water_quality_7d_avg'] = environmental_df.groupby('timestamp')['water_quality_index'].mean().rolling(7).mean()
            features['ph_7d_avg'] = environmental_df.groupby('timestamp')['ph'].mean().rolling(7).mean()
            features['turbidity_7d_avg'] = environmental_df.groupby('timestamp')['turbidity'].mean().rolling(7).mean()
            features['temperature_7d_avg'] = environmental_df.groupby('timestamp')['temperature'].mean().rolling(7).mean()
        
        # Contextual features
        if not contextual_df.empty:
            features['population_density'] = contextual_df['density_per_km2'].mean()
            features['sanitation_index'] = contextual_df['overall_sanitation_index'].mean()
            features['rainfall_7d_total'] = contextual_df['rainfall_7d'].sum()
            features['flood_risk'] = contextual_df['flood_risk'].mean()
            features['festival_flag'] = contextual_df['festival_period'].any()
            features['disaster_flag'] = contextual_df['natural_disaster'].any()
        
        # Geographic features
        if not contextual_df.empty:
            features['latitude'] = contextual_df['latitude'].mean()
            features['longitude'] = contextual_df['longitude'].mean()
            features['altitude'] = contextual_df['altitude'].mean()
        
        # Create DataFrame
        feature_df = pd.DataFrame(features)
        
        # Fill missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        # Remove infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)
        
        logger.info(f"Created tabular features: {feature_df.shape}")
        return feature_df
    
    def create_timeseries_features(self, environmental_df: pd.DataFrame,
                                 clinical_df: pd.DataFrame) -> List[TimeSeriesWindow]:
        """
        Create time-series features for LSTM model.
        
        Parameters:
        -----------
        environmental_df : pd.DataFrame
            Environmental sensor data
        clinical_df : pd.DataFrame
            Clinical case data
            
        Returns:
        --------
        List[TimeSeriesWindow]
            Time-series windows for LSTM model
        """
        logger.info("Creating time-series features...")
        
        windows = []
        sequence_length = 30  # 30-day windows
        
        if environmental_df.empty:
            return windows
        
        # Sort by timestamp
        environmental_df = environmental_df.sort_values('timestamp')
        
        # Create sliding windows
        for i in range(len(environmental_df) - sequence_length + 1):
            window_data = environmental_df.iloc[i:i + sequence_length]
            
            # Extract sensor readings
            sensor_readings = []
            case_counts = []
            environmental_factors = []
            timestamps = []
            
            for _, row in window_data.iterrows():
                # Sensor readings: [ph, turbidity, temperature, conductivity, quality_score]
                sensor_readings.append([
                    row.get('ph', 7.0),
                    row.get('turbidity', 1.0),
                    row.get('temperature', 25.0),
                    row.get('conductivity', 500.0),
                    row.get('water_quality_index', 50.0)
                ])
                
                # Case counts (would need to be joined with clinical data)
                case_counts.append(0)  # Placeholder
                
                # Environmental factors: [rainfall, humidity, air_quality]
                environmental_factors.append([
                    row.get('rainfall', 0.0),
                    row.get('humidity', 60.0),
                    row.get('air_quality_index', 50.0)
                ])
                
                timestamps.append(row['timestamp'])
            
            # Create time-series window
            window = TimeSeriesWindow(
                timestamps=timestamps,
                sensor_readings=sensor_readings,
                case_counts=case_counts,
                environmental_factors=environmental_factors,
                window_size=sequence_length
            )
            
            windows.append(window)
        
        logger.info(f"Created {len(windows)} time-series windows")
        return windows
    
    def create_text_features(self, clinical_df: pd.DataFrame) -> List[TextFeatures]:
        """
        Create text features for multilingual model.
        
        Parameters:
        -----------
        clinical_df : pd.DataFrame
            Clinical data with text fields
            
        Returns:
        --------
        List[TextFeatures]
            Text features for multilingual model
        """
        logger.info("Creating text features...")
        
        text_features = []
        
        for _, row in clinical_df.iterrows():
            # Extract text and language
            text = row.get('symptoms_text', '')
            language = row.get('language', 'auto')
            
            # Create text features
            features = TextFeatures(
                text=text,
                language=language,
                translated_text=None,
                symptom_keywords=[],
                severity_indicators=[],
                sentiment_score=None
            )
            
            # Extract keywords if text preprocessing is enabled
            if self.config["text_preprocessing"]:
                features.symptom_keywords = self._extract_symptom_keywords(text)
                features.severity_indicators = self._extract_severity_indicators(text)
                features.sentiment_score = self._calculate_sentiment_score(text)
            
            text_features.append(features)
        
        logger.info(f"Created {len(text_features)} text features")
        return text_features
    
    def _extract_text_features(self, text: str, language: LanguageCode) -> Dict[str, Any]:
        """Extract features from clinical text."""
        if not text:
            return {}
        
        # Basic text features
        features = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(text.split('.')),
            "has_numbers": bool(re.search(r'\d', text)),
            "has_special_chars": bool(re.search(r'[^\w\s]', text))
        }
        
        # Language-specific features
        if language == LanguageCode.ENGLISH:
            features["english_ratio"] = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
        elif language == LanguageCode.BENGALI:
            features["bengali_ratio"] = len(re.findall(r'[\u0980-\u09FF]', text)) / max(len(text), 1)
        elif language == LanguageCode.HINDI:
            features["hindi_ratio"] = len(re.findall(r'[\u0900-\u097F]', text)) / max(len(text), 1)
        
        return features
    
    def _extract_symptom_keywords(self, text: str) -> List[str]:
        """Extract symptom keywords from text."""
        symptom_keywords = [
            'fever', 'diarrhea', 'vomiting', 'headache', 'cough', 'cold', 'flu',
            'infection', 'pain', 'swelling', 'bleeding', 'weakness', 'fatigue'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in symptom_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_severity_indicators(self, text: str) -> List[str]:
        """Extract severity indicators from text."""
        severity_indicators = [
            'severe', 'critical', 'emergency', 'urgent', 'acute', 'chronic',
            'high', 'intense', 'extreme', 'dangerous', 'life-threatening'
        ]
        
        found_indicators = []
        text_lower = text.lower()
        for indicator in severity_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        return found_indicators
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for text."""
        positive_words = ['good', 'better', 'improved', 'recovered', 'well']
        negative_words = ['bad', 'worse', 'severe', 'critical', 'emergency']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words
    
    def _categorize_ph(self, ph: float) -> str:
        """Categorize pH value."""
        if ph < 6.5:
            return "acidic"
        elif ph > 8.5:
            return "alkaline"
        else:
            return "neutral"
    
    def _categorize_turbidity(self, turbidity: float) -> str:
        """Categorize turbidity value."""
        if turbidity < 1.0:
            return "clear"
        elif turbidity < 5.0:
            return "slightly_turbid"
        else:
            return "turbid"
    
    def _categorize_temperature(self, temperature: float) -> str:
        """Categorize temperature value."""
        if temperature < 20:
            return "cold"
        elif temperature > 30:
            return "hot"
        else:
            return "normal"
    
    def _calculate_water_quality_index(self, reading: WaterQualityReading) -> float:
        """Calculate water quality index."""
        # Simple water quality index calculation
        ph_score = 100 - abs(reading.ph - 7.0) * 10
        turbidity_score = max(0, 100 - reading.turbidity * 20)
        temperature_score = 100 - abs(reading.temperature - 25) * 2
        
        # Weighted average
        quality_index = (ph_score * 0.4 + turbidity_score * 0.4 + temperature_score * 0.2)
        return max(0, min(100, quality_index))
    
    def _calculate_contamination_risk(self, reading: WaterQualityReading) -> float:
        """Calculate contamination risk score."""
        risk_factors = []
        
        # pH risk
        if reading.ph < 6.5 or reading.ph > 8.5:
            risk_factors.append(0.3)
        
        # Turbidity risk
        if reading.turbidity > 5.0:
            risk_factors.append(0.4)
        
        # Bacterial contamination
        if reading.bacterial_test_result is False:
            risk_factors.append(0.5)
        
        # Chlorine residual
        if reading.chlorine_residual and reading.chlorine_residual < 0.2:
            risk_factors.append(0.2)
        
        return sum(risk_factors) if risk_factors else 0.0
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        if df.empty:
            return df
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if self.config["missing_value_strategy"] == "knn":
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                strategy = self.config["missing_value_strategy"]
                imputer = SimpleImputer(strategy=strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            df[categorical_cols] = df[categorical_cols].fillna("unknown")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        if df.empty:
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.config["outlier_method"] == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
            
            elif self.config["outlier_method"] == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df[col] = df[col].where(z_scores < self.config["outlier_threshold"], df[col].median())
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                known_categories = set(self.encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_categories else "unknown")
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date columns."""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_dayofyear'] = df[col].dt.dayofyear
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            df[f'{col}_is_monsoon'] = df[col].dt.month.isin([6, 7, 8, 9]).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        # Population density and sanitation interaction
        if 'density_per_km2' in df.columns and 'overall_sanitation_index' in df.columns:
            df['density_sanitation_interaction'] = df['density_per_km2'] * df['overall_sanitation_index']
        
        # Rainfall and temperature interaction
        if 'rainfall_7d' in df.columns and 'temperature_avg' in df.columns:
            df['rainfall_temperature_interaction'] = df['rainfall_7d'] * df['temperature_avg']
        
        # Water quality and population interaction
        if 'water_quality_index' in df.columns and 'density_per_km2' in df.columns:
            df['water_quality_population_interaction'] = df['water_quality_index'] * df['density_per_km2']
        
        return df
    
    def fit_preprocessing(self, df: pd.DataFrame) -> None:
        """Fit preprocessing parameters on training data."""
        if df.empty:
            return
        
        # Fit scalers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if self.config["feature_scaling"] == "standard":
                self.scalers["standard"] = StandardScaler()
                self.scalers["standard"].fit(df[numeric_cols])
            elif self.config["feature_scaling"] == "minmax":
                self.scalers["minmax"] = MinMaxScaler()
                self.scalers["minmax"].fit(df[numeric_cols])
        
        # Fit feature selectors
        if self.config["feature_selection"] and "target" in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.feature_selectors["kbest"] = SelectKBest(
                    score_func=f_classif, 
                    k=self.config["feature_selection_k"]
                )
                self.feature_selectors["kbest"].fit(df[numeric_cols], df["target"])
        
        self.is_fitted = True
        logger.info("Preprocessing parameters fitted")
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessing parameters."""
        if not self.is_fitted:
            raise ValueError("Preprocessing parameters not fitted. Call fit_preprocessing first.")
        
        if df.empty:
            return df
        
        # Apply scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and "standard" in self.scalers:
            df[numeric_cols] = self.scalers["standard"].transform(df[numeric_cols])
        elif len(numeric_cols) > 0 and "minmax" in self.scalers:
            df[numeric_cols] = self.scalers["minmax"].transform(df[numeric_cols])
        
        # Apply feature selection
        if "kbest" in self.feature_selectors:
            selected_features = self.feature_selectors["kbest"].get_support()
            selected_cols = df.select_dtypes(include=[np.number]).columns[selected_features]
            df = df[selected_cols]
        
        return df

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_data() -> Tuple[List[ClinicalReport], List[WaterQualityReading], 
                                 List[EnvironmentalContext], List[PopulationDensity],
                                 List[SanitationIndex], List[EventFlags], List[GeographicLocation]]:
    """Create sample data for testing preprocessing."""
    
    # Sample clinical reports
    clinical_reports = [
        ClinicalReport(
            patient_id_hash="hash001",
            age=25,
            sex="male",
            symptoms_text="Patient has fever and diarrhea for 2 days",
            symptoms_structured=["fever", "diarrhea"],
            date=datetime.now(),
            facility_id="facility_001",
            language=LanguageCode.ENGLISH,
            severity="medium"
        ),
        ClinicalReport(
            patient_id_hash="hash002",
            age=45,
            sex="female",
            symptoms_text="রোগীর জ্বর এবং বমি",
            symptoms_structured=["fever", "vomiting"],
            date=datetime.now(),
            facility_id="facility_002",
            language=LanguageCode.BENGALI,
            severity="high"
        )
    ]
    
    # Sample water quality readings
    water_readings = [
        WaterQualityReading(
            ph=7.2,
            turbidity=1.5,
            temperature=25.0,
            conductivity=450.0,
            bacterial_test_result=True,
            chlorine_residual=0.5,
            location="Village A",
            timestamp=datetime.now(),
            sensor_id="sensor_001",
            quality_score=85.0
        )
    ]
    
    # Sample environmental context
    environmental_context = [
        EnvironmentalContext(
            rainfall_24h=5.0,
            rainfall_7d=25.0,
            season="monsoon",
            temperature_avg=28.0,
            humidity=75.0,
            air_quality_index=60.0,
            flood_risk=0.3,
            drought_index=0.1
        )
    ]
    
    # Sample population data
    population_data = [
        PopulationDensity(
            village_ward="Village A",
            population_count=1000,
            area_km2=2.5,
            density_per_km2=400.0,
            age_distribution={"0-18": 300, "19-65": 600, "65+": 100}
        )
    ]
    
    # Sample sanitation data
    sanitation_data = [
        SanitationIndex(
            toilet_coverage=80.0,
            waste_management=70.0,
            water_access=90.0,
            hygiene_practices=75.0,
            overall_index=78.75
        )
    ]
    
    # Sample event flags
    event_flags = [
        EventFlags(
            festival_period=False,
            natural_disaster=False,
            disease_outbreak=False,
            water_contamination=False,
            health_campaign=True,
            emergency_response=False
        )
    ]
    
    # Sample geographic locations
    locations = [
        GeographicLocation(
            village_ward="Village A",
            district="Test District",
            state="Test State",
            latitude=25.0,
            longitude=80.0,
            altitude=100.0
        )
    ]
    
    return (clinical_reports, water_readings, environmental_context, 
            population_data, sanitation_data, event_flags, locations)

def main():
    """Example usage of the data preprocessor."""
    # Create sample data
    (clinical_reports, water_readings, environmental_context,
     population_data, sanitation_data, event_flags, locations) = create_sample_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess clinical data
    clinical_df = preprocessor.preprocess_clinical_data(clinical_reports)
    print("Clinical data shape:", clinical_df.shape)
    
    # Preprocess environmental data
    environmental_df = preprocessor.preprocess_environmental_data(water_readings)
    print("Environmental data shape:", environmental_df.shape)
    
    # Preprocess contextual data
    contextual_df = preprocessor.preprocess_contextual_data(
        environmental_context, population_data, sanitation_data, event_flags, locations
    )
    print("Contextual data shape:", contextual_df.shape)
    
    # Create tabular features
    tabular_df = preprocessor.create_tabular_features(clinical_df, environmental_df, contextual_df)
    print("Tabular features shape:", tabular_df.shape)
    
    # Create time-series features
    timeseries_windows = preprocessor.create_timeseries_features(environmental_df, clinical_df)
    print("Time-series windows:", len(timeseries_windows))
    
    # Create text features
    text_features = preprocessor.create_text_features(clinical_df)
    print("Text features:", len(text_features))

if __name__ == "__main__":
    main()
