"""
Multilingual Translation System
==============================

This module provides translation capabilities for multilingual text processing
in the outbreak prediction system. It includes fallback mechanisms and
language detection for clinical text analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re
import unicodedata
from datetime import datetime

# Translation libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Translation features will be limited.")

try:
    import googletrans
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    logging.warning("Googletrans library not available. Google translation features disabled.")

from schemas.ml_models import LanguageCode

logger = logging.getLogger(__name__)

class TranslationMethod(str, Enum):
    """Available translation methods."""
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    FALLBACK = "fallback"

class MultilingualTranslator:
    """
    Multilingual translation system with fallback mechanisms.
    
    This class provides translation capabilities for clinical text in multiple
    languages, with automatic language detection and fallback to simpler
    translation methods when advanced models are not available.
    """
    
    def __init__(self, default_target_language: str = "en"):
        """
        Initialize the multilingual translator.
        
        Parameters:
        -----------
        default_target_language : str
            Default target language for translation
        """
        self.default_target_language = default_target_language
        self.translators = {}
        self.translation_models = {}
        self.fallback_patterns = self._initialize_fallback_patterns()
        
        # Initialize available translation methods
        self._initialize_translators()
        
    def _initialize_fallback_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize fallback translation patterns for common medical terms."""
        return {
            "symptoms": {
                "bn": {
                    "জ্বর": "fever",
                    "ডায়রিয়া": "diarrhea", 
                    "বমি": "vomiting",
                    "মাথাব্যথা": "headache",
                    "কাশি": "cough",
                    "সর্দি": "cold",
                    "ফ্লু": "flu",
                    "সংক্রমণ": "infection",
                    "ব্যথা": "pain",
                    "ফোলা": "swelling",
                    "রক্তপাত": "bleeding",
                    "দুর্বলতা": "weakness"
                },
                "hi": {
                    "बुखार": "fever",
                    "दस्त": "diarrhea",
                    "उल्टी": "vomiting",
                    "सिरदर्द": "headache",
                    "खांसी": "cough",
                    "जुकाम": "cold",
                    "फ्लू": "flu",
                    "संक्रमण": "infection",
                    "दर्द": "pain",
                    "सूजन": "swelling",
                    "रक्तस्राव": "bleeding",
                    "कमजोरी": "weakness"
                },
                "te": {
                    "జ్వరం": "fever",
                    "అతిసారం": "diarrhea",
                    "వాంతులు": "vomiting",
                    "తలనొప్పి": "headache",
                    "కఫం": "cough",
                    "జలుబు": "cold",
                    "ఫ్లూ": "flu",
                    "సంక్రమణ": "infection",
                    "నొప్పి": "pain",
                    "వాపు": "swelling",
                    "రక్తస్రావం": "bleeding",
                    "బలహీనత": "weakness"
                }
            },
            "severity": {
                "bn": {
                    "গুরুতর": "severe",
                    "জরুরি": "urgent",
                    "তীব্র": "acute",
                    "ক্রনিক": "chronic",
                    "উচ্চ": "high",
                    "চরম": "extreme",
                    "বিপজ্জনক": "dangerous"
                },
                "hi": {
                    "गंभीर": "severe",
                    "आपातकाल": "urgent",
                    "तीव्र": "acute",
                    "पुराना": "chronic",
                    "उच्च": "high",
                    "चरम": "extreme",
                    "खतरनाक": "dangerous"
                },
                "te": {
                    "తీవ్రమైన": "severe",
                    "అత్యవసర": "urgent",
                    "తీవ్ర": "acute",
                    "దీర్ఘకాలిక": "chronic",
                    "అధిక": "high",
                    "చరమ": "extreme",
                    "ప్రమాదకర": "dangerous"
                }
            }
        }
    
    def _initialize_translators(self):
        """Initialize available translation methods."""
        # Initialize Hugging Face translation models
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load multilingual translation models
                self.translation_models["huggingface"] = {
                    "en-bn": pipeline("translation", model="Helsinki-NLP/opus-mt-en-bn"),
                    "bn-en": pipeline("translation", model="Helsinki-NLP/opus-mt-bn-en"),
                    "en-hi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
                    "hi-en": pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en"),
                    "en-te": pipeline("translation", model="Helsinki-NLP/opus-mt-en-te"),
                    "te-en": pipeline("translation", model="Helsinki-NLP/opus-mt-te-en")
                }
                logger.info("Hugging Face translation models loaded")
            except Exception as e:
                logger.warning(f"Failed to load Hugging Face models: {e}")
        
        # Initialize Google Translate
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translators["google"] = Translator()
                logger.info("Google Translate initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translate: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Parameters:
        -----------
        text : str
            Input text to analyze
            
        Returns:
        --------
        str
            Detected language code
        """
        if not text or not text.strip():
            return "en"
        
        # Clean text
        text = text.strip()
        
        # Check for specific language patterns
        language_patterns = {
            "bn": r'[\u0980-\u09FF]',  # Bengali
            "hi": r'[\u0900-\u097F]',  # Hindi
            "te": r'[\u0C00-\u0C7F]',  # Telugu
            "as": r'[\u0980-\u09FF]',  # Assamese (similar to Bengali)
        }
        
        # Count characters for each language
        language_scores = {}
        for lang, pattern in language_patterns.items():
            matches = len(re.findall(pattern, text))
            language_scores[lang] = matches
        
        # Return language with highest score
        if language_scores:
            detected_lang = max(language_scores, key=language_scores.get)
            if language_scores[detected_lang] > 0:
                return detected_lang
        
        # Default to English if no specific language detected
        return "en"
    
    def translate_text(self, text: str, target_language: str = "en", 
                      source_language: Optional[str] = None,
                      method: TranslationMethod = TranslationMethod.HUGGINGFACE) -> Dict[str, Any]:
        """
        Translate text to target language.
        
        Parameters:
        -----------
        text : str
            Text to translate
        target_language : str
            Target language code
        source_language : Optional[str]
            Source language code (auto-detect if None)
        method : TranslationMethod
            Translation method to use
            
        Returns:
        --------
        Dict[str, Any]
            Translation result with metadata
        """
        if not text or not text.strip():
            return {
                "translated_text": text,
                "source_language": source_language or "en",
                "target_language": target_language,
                "method": "none",
                "confidence": 1.0,
                "success": True
            }
        
        # Auto-detect source language if not provided
        if source_language is None:
            source_language = self.detect_language(text)
        
        # If source and target are the same, return original text
        if source_language == target_language:
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "method": "none",
                "confidence": 1.0,
                "success": True
            }
        
        # Try specified method first
        result = self._translate_with_method(text, source_language, target_language, method)
        
        # If translation failed, try fallback methods
        if not result["success"]:
            logger.warning(f"Translation with {method} failed, trying fallback methods")
            
            # Try other methods in order of preference
            fallback_methods = [
                TranslationMethod.GOOGLE,
                TranslationMethod.FALLBACK
            ]
            
            for fallback_method in fallback_methods:
                if fallback_method != method:
                    result = self._translate_with_method(text, source_language, target_language, fallback_method)
                    if result["success"]:
                        break
        
        return result
    
    def _translate_with_method(self, text: str, source_lang: str, target_lang: str, 
                              method: TranslationMethod) -> Dict[str, Any]:
        """Translate text using specified method."""
        try:
            if method == TranslationMethod.HUGGINGFACE:
                return self._translate_huggingface(text, source_lang, target_lang)
            elif method == TranslationMethod.GOOGLE:
                return self._translate_google(text, source_lang, target_lang)
            elif method == TranslationMethod.FALLBACK:
                return self._translate_fallback(text, source_lang, target_lang)
            else:
                raise ValueError(f"Unknown translation method: {method}")
        except Exception as e:
            logger.error(f"Translation with {method} failed: {e}")
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "method": method.value,
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _translate_huggingface(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate using Hugging Face models."""
        if not TRANSFORMERS_AVAILABLE or "huggingface" not in self.translation_models:
            raise Exception("Hugging Face models not available")
        
        # Get translation model
        model_key = f"{source_lang}-{target_lang}"
        if model_key not in self.translation_models["huggingface"]:
            raise Exception(f"Translation model {model_key} not available")
        
        translator = self.translation_models["huggingface"][model_key]
        
        # Translate text
        result = translator(text, max_length=512, truncation=True)
        translated_text = result[0]["translation_text"]
        
        return {
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "method": "huggingface",
            "confidence": 0.9,  # High confidence for HF models
            "success": True
        }
    
    def _translate_google(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate using Google Translate."""
        if not GOOGLETRANS_AVAILABLE or "google" not in self.translators:
            raise Exception("Google Translate not available")
        
        translator = self.translators["google"]
        
        # Translate text
        result = translator.translate(text, src=source_lang, dest=target_lang)
        translated_text = result.text
        
        # Calculate confidence based on Google's confidence score
        confidence = getattr(result, 'confidence', 0.8)
        
        return {
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "method": "google",
            "confidence": confidence,
            "success": True
        }
    
    def _translate_fallback(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate using fallback pattern matching."""
        if target_lang != "en":
            # Only support translation to English in fallback mode
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "method": "fallback",
                "confidence": 0.3,
                "success": False,
                "error": "Fallback translation only supports English target"
            }
        
        translated_text = text
        
        # Apply symptom keyword translations
        if source_lang in self.fallback_patterns["symptoms"]:
            symptom_dict = self.fallback_patterns["symptoms"][source_lang]
            for original, translation in symptom_dict.items():
                translated_text = translated_text.replace(original, translation)
        
        # Apply severity indicator translations
        if source_lang in self.fallback_patterns["severity"]:
            severity_dict = self.fallback_patterns["severity"][source_lang]
            for original, translation in severity_dict.items():
                translated_text = translated_text.replace(original, translation)
        
        # Calculate confidence based on number of translations applied
        original_words = len(text.split())
        translated_words = len(translated_text.split())
        confidence = min(0.7, 0.3 + (0.4 * (1 - abs(original_words - translated_words) / max(original_words, 1))))
        
        return {
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "method": "fallback",
            "confidence": confidence,
            "success": True
        }
    
    def batch_translate(self, texts: List[str], target_language: str = "en",
                       source_language: Optional[str] = None,
                       method: TranslationMethod = TranslationMethod.HUGGINGFACE) -> List[Dict[str, Any]]:
        """
        Translate multiple texts in batch.
        
        Parameters:
        -----------
        texts : List[str]
            List of texts to translate
        target_language : str
            Target language code
        source_language : Optional[str]
            Source language code
        method : TranslationMethod
            Translation method to use
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of translation results
        """
        results = []
        
        for text in texts:
            result = self.translate_text(text, target_language, source_language, method)
            results.append(result)
        
        return results
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """
        Get list of supported languages for each translation method.
        
        Returns:
        --------
        Dict[str, List[str]]
            Supported languages by method
        """
        supported = {
            "huggingface": ["en", "bn", "hi", "te"],
            "google": ["en", "bn", "hi", "te", "as", "ur", "ta", "gu", "pa", "or"],
            "fallback": ["en"]  # Only supports translation to English
        }
        
        return supported
    
    def get_translation_quality_score(self, original_text: str, translated_text: str) -> float:
        """
        Calculate translation quality score.
        
        Parameters:
        -----------
        original_text : str
            Original text
        translated_text : str
            Translated text
            
        Returns:
        --------
        float
            Quality score (0-1)
        """
        if not original_text or not translated_text:
            return 0.0
        
        # Basic quality metrics
        original_length = len(original_text.split())
        translated_length = len(translated_text.split())
        
        # Length ratio (should be similar)
        length_ratio = min(original_length, translated_length) / max(original_length, translated_length)
        
        # Character diversity (translated text should have reasonable character diversity)
        char_diversity = len(set(translated_text.lower())) / len(translated_text) if translated_text else 0
        
        # Word overlap with medical terms
        medical_terms = ["fever", "diarrhea", "pain", "infection", "severe", "patient", "symptoms"]
        translated_lower = translated_text.lower()
        medical_overlap = sum(1 for term in medical_terms if term in translated_lower) / len(medical_terms)
        
        # Combined quality score
        quality_score = (length_ratio * 0.4 + char_diversity * 0.3 + medical_overlap * 0.3)
        
        return min(1.0, max(0.0, quality_score))
    
    def preprocess_text_for_translation(self, text: str) -> str:
        """
        Preprocess text before translation.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        str
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove special characters but keep language-specific characters
        text = re.sub(r'[^\w\s\u0980-\u09FF\u0900-\u097F\u0C00-\u0C7F]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def postprocess_translated_text(self, text: str) -> str:
        """
        Postprocess translated text.
        
        Parameters:
        -----------
        text : str
            Translated text
            
        Returns:
        --------
        str
            Postprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common translation artifacts
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'([,.!?])\s*([,.!?])', r'\1', text)  # Remove duplicate punctuation
        
        return text.strip()

# ============================================================================
# GLOBAL TRANSLATOR INSTANCE
# ============================================================================

# Global translator instance
_translator_instance = None

def get_translator() -> MultilingualTranslator:
    """Get the global translator instance."""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = MultilingualTranslator()
    return _translator_instance

def translate_text(text: str, target_language: str = "en", 
                   source_language: Optional[str] = None,
                   method: TranslationMethod = TranslationMethod.HUGGINGFACE) -> Dict[str, Any]:
    """
    Convenience function for text translation.
    
    Parameters:
    -----------
    text : str
        Text to translate
    target_language : str
        Target language code
    source_language : Optional[str]
        Source language code
    method : TranslationMethod
        Translation method to use
        
    Returns:
    --------
    Dict[str, Any]
        Translation result
    """
    translator = get_translator()
    return translator.translate_text(text, target_language, source_language, method)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_english_text(text: str) -> bool:
    """Check if text is primarily in English."""
    if not text:
        return True
    
    # Count English characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF\u0900-\u097F\u0C00-\u0C7F]', text))
    
    if total_chars == 0:
        return True
    
    return english_chars / total_chars > 0.7

def get_language_name(language_code: str) -> str:
    """Get human-readable language name."""
    language_names = {
        "en": "English",
        "bn": "Bengali",
        "hi": "Hindi", 
        "te": "Telugu",
        "as": "Assamese",
        "ur": "Urdu",
        "ta": "Tamil",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "or": "Odia"
    }
    return language_names.get(language_code, language_code.upper())

def main():
    """Example usage of the translation system."""
    translator = MultilingualTranslator()
    
    # Test language detection
    test_texts = [
        "Patient has fever and diarrhea",
        "রোগীর জ্বর এবং ডায়রিয়া",
        "रोगी को बुखार और दस्त है",
        "రోగికి జ్వరం మరియు అతిసారం"
    ]
    
    for text in test_texts:
        detected_lang = translator.detect_language(text)
        print(f"Text: {text}")
        print(f"Detected language: {detected_lang}")
        
        # Translate to English
        result = translator.translate_text(text, "en", detected_lang)
        print(f"Translation: {result['translated_text']}")
        print(f"Method: {result['method']}, Confidence: {result['confidence']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
