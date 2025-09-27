"""
AI Assistant with LangChain Integration
======================================

This module provides AI assistant capabilities using LangChain and HuggingFace models
for multilingual health assistance, symptom analysis, and outbreak prediction explanations.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

# LangChain imports
try:
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import BaseOutputParser
    from langchain.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. AI assistant features will be limited.")

# Transformers imports
try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, TextGenerationPipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. AI assistant features will be limited.")

from schemas.ml_models import LanguageCode, InferenceResponse

logger = logging.getLogger(__name__)

class HealthAIAssistant:
    """
    AI Assistant for health-related queries using LangChain and HuggingFace models.
    
    This class provides multilingual health assistance, symptom analysis,
    and outbreak prediction explanations using LLM models.
    """
    
    def __init__(self, model_name: str = "bigscience/bloom-560m", 
                 device: str = "cpu", max_length: int = 200):
        """
        Initialize the AI assistant.
        
        Parameters:
        -----------
        model_name : str
            HuggingFace model name for text generation
        device : str
            Device to run the model on (cpu/cuda)
        max_length : int
            Maximum length for generated text
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.llm = None
        self.chains = {}
        self.tools = []
        self.agent = None
        self.memory = ConversationBufferMemory()
        
        # Initialize models if available
        if LANGCHAIN_AVAILABLE and TRANSFORMERS_AVAILABLE:
            self._initialize_models()
            self._create_tools()
            self._create_chains()
            self._create_agent()
    
    def _initialize_models(self):
        """Initialize the LLM models."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("AI Assistant models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.llm = None
    
    def _create_tools(self):
        """Create LangChain tools for the AI assistant."""
        if not self.llm:
            return
        
        # Translation tool
        translation_tool = Tool(
            name="translate_text",
            description="Translate text between languages",
            func=self._translate_text
        )
        
        # Database query tool
        database_tool = Tool(
            name="query_outbreak_history",
            description="Query historical outbreak data from database",
            func=self._query_outbreak_history
        )
        
        # IoT sensor tool
        sensor_tool = Tool(
            name="get_sensor_data",
            description="Get real-time data from IoT sensors",
            func=self._get_sensor_data
        )
        
        # Health advice tool
        health_advice_tool = Tool(
            name="get_health_advice",
            description="Get health advice based on symptoms",
            func=self._get_health_advice
        )
        
        self.tools = [
            translation_tool,
            database_tool,
            sensor_tool,
            health_advice_tool
        ]
    
    def _create_chains(self):
        """Create LangChain chains for different tasks."""
        if not self.llm:
            return
        
        # Symptom analysis chain
        symptom_template = """
        You are a health assistant. 
        The user reports: {symptoms}.
        Based on this, suggest if it could be a water-borne disease, 
        and explain in {language}.
        
        Consider:
        - Common water-borne diseases (cholera, dysentery, typhoid)
        - Severity indicators
        - Recommended actions
        - When to seek medical help
        
        Response in {language}:
        """
        
        symptom_prompt = PromptTemplate(
            input_variables=["symptoms", "language"],
            template=symptom_template
        )
        
        self.chains["symptom_analysis"] = LLMChain(
            llm=self.llm,
            prompt=symptom_prompt
        )
        
        # Outbreak explanation chain
        outbreak_template = """
        You are a public health expert.
        An outbreak prediction model has identified: {prediction_data}.
        Explain this to community health workers in {language}.
        
        Include:
        - What the prediction means
        - Risk factors identified
        - Recommended preventive measures
        - Community actions needed
        
        Response in {language}:
        """
        
        outbreak_prompt = PromptTemplate(
            input_variables=["prediction_data", "language"],
            template=outbreak_template
        )
        
        self.chains["outbreak_explanation"] = LLMChain(
            llm=self.llm,
            prompt=outbreak_prompt
        )
        
        # Health awareness chain
        awareness_template = """
        Create a health awareness message about {topic} for {community}.
        Language: {language}
        Audience: {audience}
        
        Include:
        - Key health facts
        - Prevention tips
        - Warning signs
        - When to seek help
        
        Make it engaging and culturally appropriate.
        Response in {language}:
        """
        
        awareness_prompt = PromptTemplate(
            input_variables=["topic", "community", "language", "audience"],
            template=awareness_template
        )
        
        self.chains["health_awareness"] = LLMChain(
            llm=self.llm,
            prompt=awareness_prompt
        )
    
    def _create_agent(self):
        """Create a ReAct agent with tools."""
        if not self.llm or not self.tools:
            return
        
        try:
            # Create agent prompt
            agent_prompt = """
            You are a health assistant AI. You have access to various tools to help with health-related queries.
            
            Available tools:
            {tools}
            
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Begin!
            
            Question: {input}
            Thought: {agent_scratchpad}
            """
            
            # Create agent
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=3
            )
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            self.agent_executor = None
    
    def analyze_symptoms(self, symptoms: str, language: str = "English") -> Dict[str, Any]:
        """
        Analyze symptoms and provide health advice.
        
        Parameters:
        -----------
        symptoms : str
            Reported symptoms
        language : str
            Language for response
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        if not self.chains.get("symptom_analysis"):
            return {"error": "Symptom analysis not available"}
        
        try:
            response = self.chains["symptom_analysis"].run({
                "symptoms": symptoms,
                "language": language
            })
            
            return {
                "analysis": response,
                "language": language,
                "timestamp": datetime.utcnow().isoformat(),
                "symptoms": symptoms
            }
            
        except Exception as e:
            logger.error(f"Symptom analysis failed: {e}")
            return {"error": str(e)}
    
    def explain_outbreak_prediction(self, prediction_data: Dict[str, Any], 
                                  language: str = "English") -> Dict[str, Any]:
        """
        Explain outbreak prediction results.
        
        Parameters:
        -----------
        prediction_data : Dict[str, Any]
            Prediction results from ML models
        language : str
            Language for explanation
            
        Returns:
        --------
        Dict[str, Any]
            Explanation results
        """
        if not self.chains.get("outbreak_explanation"):
            return {"error": "Outbreak explanation not available"}
        
        try:
            # Format prediction data for explanation
            formatted_data = self._format_prediction_data(prediction_data)
            
            response = self.chains["outbreak_explanation"].run({
                "prediction_data": formatted_data,
                "language": language
            })
            
            return {
                "explanation": response,
                "language": language,
                "timestamp": datetime.utcnow().isoformat(),
                "prediction_data": prediction_data
            }
            
        except Exception as e:
            logger.error(f"Outbreak explanation failed: {e}")
            return {"error": str(e)}
    
    def create_health_awareness_message(self, topic: str, community: str,
                                      language: str = "English", 
                                      audience: str = "general") -> Dict[str, Any]:
        """
        Create health awareness messages.
        
        Parameters:
        -----------
        topic : str
            Health topic for awareness
        community : str
            Target community
        language : str
            Language for message
        audience : str
            Target audience (general, health_workers, children)
            
        Returns:
        --------
        Dict[str, Any]
            Awareness message
        """
        if not self.chains.get("health_awareness"):
            return {"error": "Health awareness not available"}
        
        try:
            response = self.chains["health_awareness"].run({
                "topic": topic,
                "community": community,
                "language": language,
                "audience": audience
            })
            
            return {
                "message": response,
                "topic": topic,
                "community": community,
                "language": language,
                "audience": audience,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health awareness creation failed: {e}")
            return {"error": str(e)}
    
    def chat_with_agent(self, query: str) -> Dict[str, Any]:
        """
        Chat with the AI agent using all available tools.
        
        Parameters:
        -----------
        query : str
            User query
            
        Returns:
        --------
        Dict[str, Any]
            Agent response
        """
        if not self.agent_executor:
            return {"error": "AI agent not available"}
        
        try:
            response = self.agent_executor.run(query)
            
            return {
                "response": response,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Agent chat failed: {e}")
            return {"error": str(e)}
    
    def _translate_text(self, text: str) -> str:
        """Tool function for text translation."""
        # This would integrate with your translation system
        return f"Translated: {text}"
    
    def _query_outbreak_history(self, query: str) -> str:
        """Tool function for database queries."""
        # This would integrate with your database
        return f"Database query result for: {query}"
    
    def _get_sensor_data(self, sensor_id: str) -> str:
        """Tool function for IoT sensor data."""
        # This would integrate with your sensor APIs
        return f"Sensor data for {sensor_id}: Normal readings"
    
    def _get_health_advice(self, symptoms: str) -> str:
        """Tool function for health advice."""
        # This would integrate with your health knowledge base
        return f"Health advice for symptoms: {symptoms}"
    
    def _format_prediction_data(self, prediction_data: Dict[str, Any]) -> str:
        """Format prediction data for explanation."""
        if isinstance(prediction_data, dict):
            if "outbreak_probability" in prediction_data:
                return f"""
                Outbreak Probability: {prediction_data['outbreak_probability']:.2%}
                Confidence: {prediction_data.get('confidence', 0):.2%}
                Lead Time: {prediction_data.get('lead_time_days', 0)} days
                Severity: {prediction_data.get('severity_level', 'unknown')}
                Contributing Factors: {', '.join(prediction_data.get('contributing_factors', []))}
                """
            else:
                return str(prediction_data)
        else:
            return str(prediction_data)
    
    def get_assistant_info(self) -> Dict[str, Any]:
        """Get information about the AI assistant."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "llm_loaded": self.llm is not None,
            "chains_available": list(self.chains.keys()),
            "tools_available": len(self.tools),
            "agent_available": self.agent_executor is not None
        }

# Global AI assistant instance
_ai_assistant = None

def get_ai_assistant() -> HealthAIAssistant:
    """Get the global AI assistant instance."""
    global _ai_assistant
    if _ai_assistant is None:
        _ai_assistant = HealthAIAssistant()
    return _ai_assistant

def analyze_symptoms(symptoms: str, language: str = "English") -> Dict[str, Any]:
    """Convenience function for symptom analysis."""
    assistant = get_ai_assistant()
    return assistant.analyze_symptoms(symptoms, language)

def explain_outbreak_prediction(prediction_data: Dict[str, Any], 
                              language: str = "English") -> Dict[str, Any]:
    """Convenience function for outbreak explanation."""
    assistant = get_ai_assistant()
    return assistant.explain_outbreak_prediction(prediction_data, language)

def create_health_awareness_message(topic: str, community: str,
                                  language: str = "English", 
                                  audience: str = "general") -> Dict[str, Any]:
    """Convenience function for health awareness."""
    assistant = get_ai_assistant()
    return assistant.create_health_awareness_message(topic, community, language, audience)

def chat_with_agent(query: str) -> Dict[str, Any]:
    """Convenience function for agent chat."""
    assistant = get_ai_assistant()
    return assistant.chat_with_agent(query)
