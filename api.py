#!/usr/bin/env python3
"""
Entity-Level Sentiment Analysis REST API

This module provides a FastAPI REST service for entity-level sentiment analysis.
It segments multi-paragraph text into sentences, processes them, and returns
structured sentiment results for entities detected within the text.
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import nltk
from nltk.tokenize import sent_tokenize
import uvicorn

from entity_sentiment import EntitySentimentAnalyzer

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources for sentence segmentation
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}. Sentence segmentation may be affected.")

# Initialize the FastAPI app
app = FastAPI(
    title="Entity-Level Sentiment Analysis API",
    description="API for analyzing sentiment associated with entities in text",
    version="1.0.0"
)

# Create an instance of the sentiment analyzer
try:
    analyzer = EntitySentimentAnalyzer()
    logger.info("Successfully initialized sentiment analyzer")
except Exception as e:
    logger.error(f"Failed to initialize sentiment analyzer: {e}")
    analyzer = None  # We'll check this later and return an appropriate error


# Define request and response models
class SentimentRequest(BaseModel):
    text: str = Field(..., description="The text to analyze", min_length=1)
    batch_size: Optional[int] = Field(10, description="Batch size for processing multiple sentences")
    include_context: Optional[bool] = Field(False, description="Whether to include context information in the response")


class EntitySentiment(BaseModel):
    entity: str = Field(..., description="The entity text")
    entity_type: str = Field(..., description="The type of entity")
    sentiment: Dict[str, Any] = Field(..., description="Sentiment information including score and label")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="Context information if requested")
    contrasting_details: Optional[List[Dict[str, Any]]] = Field(None, description="Contrasting sentiment details if present")
    fallback: Optional[bool] = Field(None, description="Whether this is a fallback result")
    warning: Optional[str] = Field(None, description="Warning message if applicable")


class SentenceSentiment(BaseModel):
    text: str = Field(..., description="The sentence text")
    overall_sentiment: Dict[str, Any] = Field(..., description="Overall sentiment for the sentence")
    entity_sentiments: List[EntitySentiment] = Field(..., description="Sentiments for entities found in the sentence")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


class SentimentResponse(BaseModel):
    text: str = Field(..., description="The original input text")
    sentences: List[SentenceSentiment] = Field(..., description="Analysis for each sentence")
    overall_sentiment: Dict[str, Any] = Field(..., description="Overall sentiment for the entire text")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


def segment_text(text: str) -> List[str]:
    """
    Segment text into sentences.
    
    Args:
        text: Input text to segment
        
    Returns:
        List of sentences
    """
    try:
        sentences = sent_tokenize(text)
        return sentences
    except Exception as e:
        logger.error(f"Error in sentence segmentation: {e}")
        # Fallback to simple splitting
        simple_sentences = [s.strip() for s in text.split('.') if s.strip()]
        if simple_sentences:
            return simple_sentences
        else:
            # If all else fails, return the whole text as one sentence
            return [text]


def process_in_batches(sentences: List[str], batch_size: int, include_context: bool) -> List[SentenceSentiment]:
    """
    Process sentences in batches to avoid memory issues with large texts.
    
    Args:
        sentences: List of sentences to process
        batch_size: Number of sentences to process in each batch
        include_context: Whether to include context information
        
    Returns:
        List of SentenceSentiment objects
    """
    results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        for sentence in batch:
            try:
                # Analyze the sentence
                analysis = analyzer.analyze_with_explanation(sentence)
                
                # Filter out context information if not requested
                if not include_context:
                    for entity in analysis["entity_sentiments"]:
                        if "contexts" in entity:
                            del entity["contexts"]
                
                sentence_result = SentenceSentiment(
                    text=sentence,
                    overall_sentiment=analysis["overall_sentiment"],
                    entity_sentiments=analysis["entity_sentiments"],
                    error=None
                )
                
                results.append(sentence_result)
            except Exception as e:
                logger.error(f"Error processing sentence '{sentence}': {e}")
                # Add an error entry
                error_result = SentenceSentiment(
                    text=sentence,
                    overall_sentiment={"score": 0.5, "label": "neutral"},
                    entity_sentiments=[],
                    error=str(e)
                )
                results.append(error_result)
    
    return results


def calculate_overall_sentiment(sentence_results: List[SentenceSentiment]) -> Dict[str, Any]:
    """
    Calculate overall sentiment for the entire text based on individual sentence sentiments.
    
    Args:
        sentence_results: List of sentence sentiment results
        
    Returns:
        Overall sentiment dictionary
    """
    if not sentence_results:
        return {"score": 0.5, "label": "neutral"}
    
    # Calculate weighted average of scores based on sentence length
    total_length = sum(len(s.text) for s in sentence_results)
    
    if total_length == 0:
        return {"score": 0.5, "label": "neutral"}
    
    weighted_score = sum(s.overall_sentiment["score"] * len(s.text) / total_length for s in sentence_results)
    
    # Determine sentiment label based on weighted score
    if weighted_score > 0.75:
        label = "very positive"
    elif weighted_score > 0.6:
        label = "positive"
    elif weighted_score > 0.45:
        label = "slightly positive"
    elif weighted_score > 0.35:
        label = "neutral"
    elif weighted_score > 0.2:
        label = "slightly negative" 
    elif weighted_score > 0.05:
        label = "negative"
    else:
        label = "very negative"
    
    return {
        "score": weighted_score,
        "label": label
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Entity-Level Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze sentiment for entities in text",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if analyzer is None:
        return {"status": "error", "message": "Sentiment analyzer failed to initialize"}
    return {"status": "ok", "message": "API is operational"}


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment for entities in the provided text.
    
    The text is segmented into sentences, and each sentence is analyzed separately.
    Entity-level sentiment is extracted for each recognized entity.
    """
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Sentiment analyzer is not available")
    
    try:
        # Get the text and options
        text = request.text
        batch_size = request.batch_size
        include_context = request.include_context
        
        # Segment the text into sentences
        sentences = segment_text(text)
        
        # Process sentences in batches
        sentence_results = process_in_batches(sentences, batch_size, include_context)
        
        # Calculate overall sentiment
        overall_sentiment = calculate_overall_sentiment(sentence_results)
        
        # Prepare the response
        response = SentimentResponse(
            text=text,
            sentences=sentence_results,
            overall_sentiment=overall_sentiment,
            error=None
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

