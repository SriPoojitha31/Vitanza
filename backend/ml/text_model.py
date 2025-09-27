"""
Multilingual Text Classifier for Outbreak Prediction
====================================================

This module implements a multilingual text classifier using XLM-Roberta
for analyzing clinical text in multiple languages to predict outbreak risk.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModel, Trainer, TrainingArguments, pipeline
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import re
from datetime import datetime

from schemas.ml_models import TextFeatures, LanguageCode, ModelEvaluation

logger = logging.getLogger(__name__)

class MultilingualTextClassifier:
    """
    Multilingual text classifier for outbreak prediction.
    
    This model uses XLM-Roberta to analyze clinical text in multiple languages
    and predict outbreak risk based on symptom descriptions and severity indicators.
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base",
                 model_path: str = "models/text_xlm_roberta",
                 max_length: int = 128):
        """
        Initialize the multilingual text classifier.
        
        Parameters:
        -----------
        model_name : str
            Hugging Face model name
        model_path : str
            Path to save/load the trained model
        max_length : int
            Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Language detection patterns
        self.language_patterns = {
            LanguageCode.ENGLISH: r'[a-zA-Z]',
            LanguageCode.BENGALI: r'[\u0980-\u09FF]',
            LanguageCode.HINDI: r'[\u0900-\u097F]',
            LanguageCode.TELUGU: r'[\u0C00-\u0C7F]',
            LanguageCode.ASSAMESE: r'[\u0980-\u09FF]'
        }
        
        # Symptom keywords in different languages
        self.symptom_keywords = {
            LanguageCode.ENGLISH: [
                'fever', 'diarrhea', 'vomiting', 'nausea', 'headache', 'cough',
                'cold', 'flu', 'infection', 'pain', 'ache', 'sore', 'rash',
                'swelling', 'bleeding', 'weakness', 'fatigue', 'chills'
            ],
            LanguageCode.BENGALI: [
                'জ্বর', 'ডায়রিয়া', 'বমি', 'মাথাব্যথা', 'কাশি', 'সর্দি',
                'ফ্লু', 'সংক্রমণ', 'ব্যথা', 'ফোলা', 'রক্তপাত', 'দুর্বলতা'
            ],
            LanguageCode.HINDI: [
                'बुखार', 'दस्त', 'उल्टी', 'सिरदर्द', 'खांसी', 'जुकाम',
                'फ्लू', 'संक्रमण', 'दर्द', 'सूजन', 'रक्तस्राव', 'कमजोरी'
            ],
            LanguageCode.TELUGU: [
                'జ్వరం', 'అతిసారం', 'వాంతులు', 'తలనొప్పి', 'కఫం', 'జలుబు',
                'ఫ్లూ', 'సంక్రమణ', 'నొప్పి', 'వాపు', 'రక్తస్రావం', 'బలహీనత'
            ]
        }
        
        # Severity indicators
        self.severity_indicators = {
            LanguageCode.ENGLISH: [
                'severe', 'critical', 'emergency', 'urgent', 'acute', 'chronic',
                'high', 'intense', 'extreme', 'dangerous', 'life-threatening'
            ],
            LanguageCode.BENGALI: [
                'গুরুতর', 'জরুরি', 'তীব্র', 'ক্রনিক', 'উচ্চ', 'চরম', 'বিপজ্জনক'
            ],
            LanguageCode.HINDI: [
                'गंभीर', 'आपातकाल', 'तीव्र', 'पुराना', 'उच्च', 'चरम', 'खतरनाक'
            ],
            LanguageCode.TELUGU: [
                'తీవ్రమైన', 'అత్యవసర', 'తీవ్ర', 'దీర్ఘకాలిక', 'అధిక', 'చరమ', 'ప్రమాదకర'
            ]
        }
    
    def detect_language(self, text: str) -> LanguageCode:
        """
        Detect the language of the input text.
        
        Parameters:
        -----------
        text : str
            Input text to analyze
            
        Returns:
        --------
        LanguageCode
            Detected language
        """
        text = text.strip()
        if not text:
            return LanguageCode.ENGLISH
        
        # Count characters for each language
        language_scores = {}
        for lang, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, text))
            language_scores[lang] = matches
        
        # Return language with highest score
        detected_lang = max(language_scores, key=language_scores.get)
        
        # If no specific language detected, default to English
        if language_scores[detected_lang] == 0:
            return LanguageCode.ENGLISH
        
        return detected_lang
    
    def extract_symptom_keywords(self, text: str, language: LanguageCode) -> List[str]:
        """
        Extract symptom keywords from text.
        
        Parameters:
        -----------
        text : str
            Input text
        language : LanguageCode
            Text language
            
        Returns:
        --------
        List[str]
            Extracted symptom keywords
        """
        if language not in self.symptom_keywords:
            language = LanguageCode.ENGLISH
        
        keywords = self.symptom_keywords[language]
        found_keywords = []
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def extract_severity_indicators(self, text: str, language: LanguageCode) -> List[str]:
        """
        Extract severity indicators from text.
        
        Parameters:
        -----------
        text : str
            Input text
        language : LanguageCode
            Text language
            
        Returns:
        --------
        List[str]
            Extracted severity indicators
        """
        if language not in self.severity_indicators:
            language = LanguageCode.ENGLISH
        
        indicators = self.severity_indicators[language]
        found_indicators = []
        
        text_lower = text.lower()
        for indicator in indicators:
            if indicator.lower() in text_lower:
                found_indicators.append(indicator)
        
        return found_indicators
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for model input.
        
        Parameters:
        -----------
        text : str
            Raw text input
            
        Returns:
        --------
        str
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep language-specific characters
        text = re.sub(r'[^\w\s\u0980-\u09FF\u0900-\u097F\u0C00-\u0C7F]', '', text)
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        return text
    
    def prepare_dataset(self, data: pd.DataFrame, 
                       text_column: str = 'text',
                       label_column: str = 'label',
                       language_column: str = 'language') -> Dataset:
        """
        Prepare dataset for training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        text_column : str
            Name of text column
        label_column : str
            Name of label column
        language_column : str
            Name of language column
            
        Returns:
        --------
        Dataset
            Hugging Face dataset
        """
        logger.info("Preparing multilingual text dataset...")
        
        # Initialize tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def tokenize_function(examples):
            # Preprocess texts
            texts = [self.preprocess_text(text) for text in examples[text_column]]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_pandas(data)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(type='torch')
        
        logger.info(f"Prepared dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(self, data: pd.DataFrame, 
              text_column: str = 'text',
              label_column: str = 'label',
              language_column: str = 'language',
              test_size: float = 0.2,
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 2e-5) -> Dict[str, Any]:
        """
        Train the multilingual text classifier.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        text_column : str
            Name of text column
        label_column : str
            Name of label column
        language_column : str
            Name of language column
        test_size : float
            Proportion of data for testing
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metrics
        """
        logger.info("Training multilingual text classifier...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_data, test_data = train_test_split(
            data, test_size=test_size, stratify=data[label_column], random_state=42
        )
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data, text_column, label_column, language_column)
        test_dataset = self.prepare_dataset(test_data, text_column, label_column, language_column)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.model_path}/logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            greater_is_better=True,
            report_to=None  # Disable wandb
        )
        
        # Custom metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1]
            labels = torch.tensor(labels)
            
            auc_score = roc_auc_score(labels, predictions)
            return {"auc": auc_score}
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_path)
        
        # Test predictions
        test_predictions = trainer.predict(test_dataset)
        test_auc = test_predictions.metrics['test_auc']
        
        results = {
            "test_auc": test_auc,
            "eval_results": eval_results,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_name": self.model_name
        }
        
        logger.info(f"Training completed. Test AUC: {test_auc:.4f}")
        return results
    
    def predict(self, text_features: TextFeatures) -> float:
        """
        Make prediction for text features.
        
        Parameters:
        -----------
        text_features : TextFeatures
            Text features for prediction
            
        Returns:
        --------
        float
            Prediction probability
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Detect language if auto
        if text_features.language == LanguageCode.AUTO:
            detected_lang = self.detect_language(text_features.text)
            text_features.language = detected_lang
        
        # Preprocess text
        processed_text = self.preprocess_text(text_features.text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = probabilities[0, 1].item()
        
        return prediction
    
    def predict_batch(self, text_features_list: List[TextFeatures]) -> List[float]:
        """
        Make predictions for multiple text features.
        
        Parameters:
        -----------
        text_features_list : List[TextFeatures]
            List of text features
            
        Returns:
        --------
        List[float]
            List of prediction probabilities
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        predictions = []
        
        for text_features in text_features_list:
            pred = self.predict(text_features)
            predictions.append(pred)
        
        return predictions
    
    def analyze_text(self, text: str, language: LanguageCode = LanguageCode.AUTO) -> Dict[str, Any]:
        """
        Comprehensive text analysis for outbreak prediction.
        
        Parameters:
        -----------
        text : str
            Input text
        language : LanguageCode
            Text language
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        # Detect language if auto
        if language == LanguageCode.AUTO:
            language = self.detect_language(text)
        
        # Create text features
        text_features = TextFeatures(
            text=text,
            language=language,
            translated_text=None,
            symptom_keywords=[],
            severity_indicators=[],
            sentiment_score=None
        )
        
        # Extract keywords
        symptom_keywords = self.extract_symptom_keywords(text, language)
        severity_indicators = self.extract_severity_indicators(text, language)
        
        # Update text features
        text_features.symptom_keywords = symptom_keywords
        text_features.severity_indicators = severity_indicators
        
        # Make prediction
        prediction = self.predict(text_features)
        
        # Calculate sentiment score (simple keyword-based)
        sentiment_score = self.calculate_sentiment_score(text, language)
        text_features.sentiment_score = sentiment_score
        
        return {
            "prediction": prediction,
            "language": language,
            "symptom_keywords": symptom_keywords,
            "severity_indicators": severity_indicators,
            "sentiment_score": sentiment_score,
            "text_length": len(text),
            "keyword_count": len(symptom_keywords),
            "severity_count": len(severity_indicators)
        }
    
    def calculate_sentiment_score(self, text: str, language: LanguageCode) -> float:
        """
        Calculate sentiment score based on keywords.
        
        Parameters:
        -----------
        text : str
            Input text
        language : LanguageCode
            Text language
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        # Simple keyword-based sentiment
        positive_keywords = ['good', 'better', 'improved', 'recovered', 'well']
        negative_keywords = ['bad', 'worse', 'severe', 'critical', 'emergency']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_keywords
        return max(-1.0, min(1.0, sentiment))
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        logger.info("Multilingual text model loaded successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
        --------
        Dict[str, Any]
            Model information
        """
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "max_length": self.max_length,
            "device": str(self.device),
            "supported_languages": list(self.language_patterns.keys()),
            "symptom_keywords_count": {lang: len(keywords) for lang, keywords in self.symptom_keywords.items()},
            "severity_indicators_count": {lang: len(indicators) for lang, indicators in self.severity_indicators.items()}
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_text_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample text data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample text data
    """
    np.random.seed(42)
    
    # Sample texts in different languages
    english_texts = [
        "Patient has severe fever and diarrhea for 3 days",
        "Mild headache and cough, no fever",
        "Critical condition with high fever and vomiting",
        "Patient recovered well from flu symptoms",
        "Acute respiratory infection with breathing difficulty"
    ]
    
    bengali_texts = [
        "রোগীর তিন দিন ধরে জ্বর এবং ডায়রিয়া",
        "মাথাব্যথা এবং কাশি, জ্বর নেই",
        "গুরুতর অবস্থা, উচ্চ জ্বর এবং বমি",
        "রোগী ফ্লু থেকে সুস্থ হয়ে উঠেছে",
        "তীব্র শ্বাসকষ্টের সংক্রমণ"
    ]
    
    hindi_texts = [
        "रोगी को तीन दिन से तेज बुखार और दस्त",
        "सिरदर्द और खांसी, बुखार नहीं",
        "गंभीर स्थिति, तेज बुखार और उल्टी",
        "रोगी फ्लू से ठीक हो गया",
        "तीव्र श्वसन संक्रमण"
    ]
    
    all_texts = english_texts + bengali_texts + hindi_texts
    languages = [LanguageCode.ENGLISH] * len(english_texts) + \
                [LanguageCode.BENGALI] * len(bengali_texts) + \
                [LanguageCode.HINDI] * len(hindi_texts)
    
    # Generate data
    texts = []
    labels = []
    lang_list = []
    
    for i in range(n_samples):
        text_idx = np.random.randint(0, len(all_texts))
        texts.append(all_texts[text_idx])
        lang_list.append(languages[text_idx])
        
        # Simple label based on keywords
        text_lower = all_texts[text_idx].lower()
        if any(word in text_lower for word in ['severe', 'critical', 'acute', 'গুরুতর', 'गंभीर', 'तीव्र']):
            labels.append(1)
        else:
            labels.append(0)
    
    data = pd.DataFrame({
        'text': texts,
        'label': labels,
        'language': lang_list
    })
    
    return data

def main():
    """Example usage of the multilingual text classifier."""
    # Create sample data
    data = create_sample_text_data(100)
    
    # Initialize model
    model = MultilingualTextClassifier()
    
    # Train model
    results = model.train(data)
    print("Training results:", results)
    
    # Test prediction
    sample_text = TextFeatures(
        text="Patient has severe fever and diarrhea",
        language=LanguageCode.ENGLISH
    )
    
    prediction = model.predict(sample_text)
    print(f"Sample prediction: {prediction}")
    
    # Test analysis
    analysis = model.analyze_text("রোগীর জ্বর এবং ডায়রিয়া", LanguageCode.BENGALI)
    print("Text analysis:", analysis)

if __name__ == "__main__":
    main()
