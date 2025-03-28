#!/usr/bin/env python3
"""
Entity-Level Sentiment Analysis

This module implements entity-level sentiment analysis by combining:
1. Named Entity Recognition (NER) using spaCy
2. Sentiment Analysis using BERT from Hugging Face

It can identify entities in text and determine the sentiment associated with each entity.
"""

import spacy
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntitySentimentAnalyzer:
    """
    A class that performs entity-level sentiment analysis by combining
    named entity recognition with sentiment analysis.
    """
    
    def __init__(self, bert_model="distilbert-base-uncased-finetuned-sst-2-english", spacy_model="en_core_web_sm"):
        """
        Initialize the EntitySentimentAnalyzer with the specified models.
        
        Args:
            bert_model (str): The BERT model to use for sentiment analysis
            spacy_model (str): The spaCy model to use for NER
        """
        logger.info(f"Initializing EntitySentimentAnalyzer with {bert_model} and {spacy_model}")
        
        # Initialize sentiment modifier lists
        # Common negation words and phrases
        self.negation_words = [
            "not", "n't", "no", "never", "none", "nothing", "neither", "nor", 
            "hardly", "barely", "scarcely", "doesn't", "isn't", "aren't", "wasn't",
            "weren't", "hasn't", "haven't", "won't", "wouldn't", "don't",
            "cannot", "can't", "couldn't", "without", "absent", "lack", "lacking", 
            "fails", "failed", "prevents", "prevented", "avoids", "avoided", "denies", 
            "denied", "rejects", "rejected", "refuse", "refuses", "refused"
        ]
        
        # Sentiment-inverting verbs
        self.inverting_verbs = [
            "hate", "hates", "hated", "hating",
            "dislike", "dislikes", "disliked", "disliking",
            "loathe", "loathes", "loathed", "loathing",
            "detest", "detests", "detested", "detesting",
            "despise", "despises", "despised", "despising",
            "abhor", "abhors", "abhorred", "abhorring",
            "resent", "resents", "resented", "resenting"
        ]
        
        # Sentiment intensifiers (boost sentiment)
        self.intensifiers = [
            "very", "extremely", "incredibly", "really", "truly", "absolutely", 
            "completely", "totally", "thoroughly", "utterly", "quite", "particularly",
            "especially", "exceptionally", "exceedingly", "immensely", "enormously",
            "deeply", "strongly", "highly", "intensely", "remarkably"
        ]
        
        # Sentiment diminishers (weaken sentiment)
        self.diminishers = [
            "somewhat", "slightly", "a bit", "a little", "kind of", "kinda", 
            "sort of", "rather", "fairly", "pretty", "relatively", "moderately",
            "marginally", "nominally", "somewhat", "partially", "barely", "hardly"
        ]
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise RuntimeError(f"Could not load spaCy model '{spacy_model}'. Make sure it's installed.") from e
        
        # Load BERT model and tokenizer for sentiment analysis
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                bert_model
            )
            logger.info("Successfully loaded BERT model and tokenizer")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise RuntimeError(f"Could not load BERT model '{bert_model}'. Check your internet connection or model name.") from e
    
    def get_entities(self, text):
        """
        Extract named entities from the text.
        
        Args:
            text (str): The input text
            
        Returns:
            list: A list of (entity, entity_type) tuples
        """
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def get_sentiment(self, text):
        """
        Analyze the sentiment of the text.
        
        Args:
            text (str): The input text
            
        Returns:
            dict: A dictionary with sentiment information including score and label
        """
        # Tokenize and prepare input for the model
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get sentiment prediction
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        logits = outputs.logits
        
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
        
        # The models I've suggested have binary labels: negative (0) and positive (1)
        sentiment_score = probabilities[1]  # Probability for positive class
        
        # Determine label based on score
        if sentiment_score > 0.6:
            label = "positive"
        elif sentiment_score < 0.4:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": float(sentiment_score),
            "label": label,
            "probabilities": {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        }
    
    def get_entity_contexts(self, text, entity, window_size=10):
        """
        Extract contexts around a specific entity in the text with improved accuracy.
        
        Args:
            text (str): The input text
            entity (str): The entity to find contexts for
            window_size (int): Base number of words to include before and after the entity
            
        Returns:
            list: A list of tuples containing context strings, their relevance weight,
                  and additional context metadata
        """
        doc = self.nlp(text)
        words = [token.text for token in doc]
        tokens = [token for token in doc]
        contexts = []
        
        entity_words = entity.split()
        entity_len = len(entity_words)
        
        # Get sentence boundaries for better context analysis
        sentences = list(doc.sents)
        sent_boundaries = [(sent.start, sent.end) for sent in sentences]
        
        # Contrasting conjunctions that often separate contrasting clauses
        contrasting_conjunctions = ["but", "however", "yet", "although", "though", "nevertheless", 
                                   "whereas", "while", "on the other hand", "in contrast"]
        
        for i in range(len(words) - entity_len + 1):
            if ' '.join(words[i:i+entity_len]).lower() == entity.lower():
                # Find which sentence contains this entity
                entity_sent_idx = None
                entity_position = i
                
                for idx, (start, end) in enumerate(sent_boundaries):
                    if start <= i < end:
                        entity_sent_idx = idx
                        break
                
                if entity_sent_idx is not None:
                    # Get sentence that contains the entity
                    sent_start, sent_end = sent_boundaries[entity_sent_idx]
                    sent_text = ' '.join(words[sent_start:sent_end])
                    
                    # Find the entity's governing verb in the dependency tree
                    entity_token_idx = None
                    governing_verb = None
                    
                    # Find main token for the entity
                    for j in range(i, i+entity_len):
                        if doc[j].pos_ in ["NOUN", "PROPN"]:
                            entity_token_idx = j
                            break
                    
                    if entity_token_idx is not None:
                        # Find governing verb by following dependency tree upward
                        current = doc[entity_token_idx]
                        # Follow the dependency tree up to find a verb
                        while current.head != current:  # Stop if we reach the root
                            if current.head.pos_ == "VERB":
                                governing_verb = current.head.text
                                break
                            current = current.head
                    
                    # Check for contrasting conjunctions within the sentence
                    clause_boundaries = []
                    
                    for j in range(sent_start, sent_end):
                        if any(conj in doc[j:min(j+3, sent_end)].text.lower() for conj in contrasting_conjunctions):
                            # Found a contrasting conjunction - split the context here
                            clause_boundaries.append(j)
                    
                    # If contrasting conjunctions found, create contexts for each clause
                    if clause_boundaries:
                        # Add sentence start as first boundary
                        boundaries = [sent_start] + clause_boundaries + [sent_end]
                        
                        # Process each clause
                        for k in range(len(boundaries) - 1):
                            clause_start = boundaries[k]
                            clause_end = boundaries[k + 1]
                            
                            # Check if entity is within this clause
                            if clause_start <= i < clause_end:
                                # Entity is in this clause - higher weight
                                context = ' '.join(words[clause_start:clause_end])
                                # First clause often has more weight in English
                                position_factor = 1.2 if k == 0 else 1.0
                                weight = 1.5 * position_factor  # Higher weight for containing clause
                                
                                # Add metadata about this context
                                context_meta = {
                                    "contains_entity": True,
                                    "clause_position": k,
                                    "governing_verb": governing_verb,
                                    "is_contrasting": False
                                }
                                
                                contexts.append((context, weight, context_meta))
                            else:
                                # Entity is not in this clause - check if it's a contrasting clause
                                context = ' '.join(words[clause_start:clause_end])
                                # Contrasting weight based on distance from entity clause
                                distance = min(abs(k - c) for c in range(len(boundaries) - 1) 
                                             if boundaries[c] <= i < boundaries[c+1])
                                
                                # Closer contrasting clauses have more relevance
                                contrast_weight = 0.9 / (1 + distance)
                                
                                # Add metadata about this context
                                context_meta = {
                                    "contains_entity": False,
                                    "clause_position": k,
                                    "governing_verb": None,  # No direct verb relation
                                    "is_contrasting": True
                                }
                                
                                contexts.append((context, contrast_weight, context_meta))
                    else:
                        # No contrasting conjunctions, use adaptive window around entity
                        
                        # Calculate adaptive window size based on sentence length
                        sent_length = sent_end - sent_start
                        adaptive_window = min(window_size, max(3, sent_length // 2))
                        
                        # Calculate asymmetric context bounds
                        # More context before entity for subject-first languages like English
                        pre_context = min(adaptive_window + 2, i - sent_start)
                        post_context = min(adaptive_window - 2, sent_end - (i + entity_len))
                        
                        start = max(sent_start, i - pre_context)
                        end = min(sent_end, i + entity_len + post_context)
                        
                        # Extract context within sentence boundaries
                        context = ' '.join(words[start:end])
                        
                        # Calculate relevance weight based on context size and position
                        # Shorter, more focused contexts get higher weights
                        context_len = end - start
                        weight = 1.0
                        if context_len <= entity_len + 4:  # Very tight context
                            weight = 1.5
                        elif context_len >= entity_len + 10:  # Wider context
                            weight = 0.8
                        
                        # Add metadata about this context
                        context_meta = {
                            "contains_entity": True,
                            "clause_position": 0,  # Single clause
                            "governing_verb": governing_verb,
                            "is_contrasting": False
                        }
                        
                        contexts.append((context, weight, context_meta))
                    
                    # Check for contrasting statements in adjacent sentences
                    if entity_sent_idx > 0:
                        prev_sent_start, prev_sent_end = sent_boundaries[entity_sent_idx - 1]
                        prev_sent_text = ' '.join(words[prev_sent_start:prev_sent_end])
                        
                        # Check if the previous sentence contains a contrasting conjunction
                        # or if there's a contrasting conjunction between sentences
                        if any(conj in prev_sent_text.lower() for conj in contrasting_conjunctions) or \
                           any(conj in words[prev_sent_end-1:sent_start+1] for conj in contrasting_conjunctions):
                            
                            prev_context = ' '.join(words[prev_sent_start:prev_sent_end])
                            
                            # Add metadata about this context
                            context_meta = {
                                "contains_entity": False,
                                "clause_position": -1,  # Previous sentence
                                "governing_verb": None,
                                "is_contrasting": True
                            }
                            
                            contexts.append((prev_context, 0.7, context_meta))  # Lower weight for adjacent sentence
                    
                    if entity_sent_idx < len(sent_boundaries) - 1:
                        next_sent_start, next_sent_end = sent_boundaries[entity_sent_idx + 1]
                        next_sent_text = ' '.join(words[next_sent_start:next_sent_end])
                        
                        # Check if the next sentence contains a contrasting conjunction
                        # or if there's a contrasting conjunction between sentences
                        if any(conj in next_sent_text.lower() for conj in contrasting_conjunctions) or \
                           any(conj in words[sent_end-1:next_sent_start+1] for conj in contrasting_conjunctions):
                            
                            next_context = ' '.join(words[next_sent_start:next_sent_end])
                            
                            # Add metadata about this context
                            context_meta = {
                                "contains_entity": False,
                                "clause_position": 1,  # Next sentence
                                "governing_verb": None,
                                "is_contrasting": True
                            }
                            
                            contexts.append((next_context, 0.7, context_meta))  # Lower weight for adjacent sentence
                
                else:
                    # Fallback to the original window method if sentence parsing fails
                    start = max(0, i - window_size)
                    end = min(len(words), i + entity_len + window_size)
                    context = ' '.join(words[start:end])
                    
                    # Basic metadata for fallback case
                    context_meta = {
                        "contains_entity": True,
                        "clause_position": 0,
                        "governing_verb": None,
                        "is_contrasting": False
                    }
                    
                    contexts.append((context, 1.0, context_meta))
        
        return contexts
    def detect_sentiment_modifiers(self, text, entity_start=None, entity_end=None):
        """
        Detect sentiment modifiers (negations, intensifiers, diminishers) in text
        and their relationship to a given entity.
        
        Args:
            text (str): The text to analyze
            entity_start (int, optional): The start token position of the entity
            entity_end (int, optional): The end token position of the entity
            
        Returns:
            dict: A dictionary with detected modifiers and their impact
        """
        doc = self.nlp(text.lower())
        modifiers = {
            "has_negation": False,
            "negation_scope": [],
            "intensifiers": [],
            "diminishers": [],
            "sentiment_impact": 1.0  # Default multiplier (no change)
        }
        
        # Process tokens to find modifiers
        for i, token in enumerate(doc):
            # Check for negations
            if token.text in self.negation_words or token.lemma_ in self.negation_words or token.text in self.inverting_verbs or token.lemma_ in self.inverting_verbs:
                modifiers["has_negation"] = True
                
                # Determine negation scope (usually 3-5 tokens after the negation)
                scope_start = i
                scope_end = min(i + 5, len(doc))
                
                # Adjust scope end to stop at sentence boundaries or conjunctions
                for j in range(i+1, scope_end):
                    if doc[j].is_punct and doc[j].text in ['.', '!', '?']:
                        scope_end = j
                        break
                    if doc[j].text in ['but', 'however', 'nevertheless', 'yet', 'although']:
                        scope_end = j
                        break
                        
                modifiers["negation_scope"].append((scope_start, scope_end))
                
                # If we know the entity position, check if it's within scope
                if entity_start is not None and entity_end is not None:
                    if (scope_start <= entity_start < scope_end) or \
                       any(doc[entity_start].head.i == j for j in range(scope_start, scope_end)):
                        modifiers["sentiment_impact"] *= -1.0
                        
            # Check for intensifiers
            elif token.text in self.intensifiers:
                modifiers["intensifiers"].append(i)
                
                # If entity is within 3 tokens of intensifier or directly related
                if entity_start is not None and entity_end is not None:
                    if abs(i - entity_start) <= 3 or doc[entity_start].head == token or token.head == doc[entity_start]:
                        # Boost sentiment strength (positive or negative)
                        modifiers["sentiment_impact"] *= 1.5
                        
            # Check for diminishers
            elif token.text in self.diminishers or (token.text.startswith("a ") and token.text.endswith(("bit", "little"))):
                modifiers["diminishers"].append(i)
                
                # If entity is within 3 tokens of diminisher or directly related
                if entity_start is not None and entity_end is not None:
                    if abs(i - entity_start) <= 3 or doc[entity_start].head == token or token.head == doc[entity_start]:
                        # Weaken sentiment strength (positive or negative)
                        modifiers["sentiment_impact"] *= 0.7
        
        return modifiers
    
    def check_for_negation(self, text, entity_position=None):
        """
        Check if the text contains negation words that might affect sentiment.
        
        Args:
            text (str): The text to check for negation
            entity_position (tuple, optional): The (start, end) position of the entity
            
        Returns:
            bool: True if negation is detected, False otherwise
        """
        
        # Check for explicit negation words
        doc = self.nlp(text.lower())
        
        # If we know where the entity is, check for negation in its local context
        if entity_position:
            start, end = entity_position
            
            # Check for negation before the entity (within a window of 5 tokens)
            pre_window = max(0, start - 5)
            pre_context = doc[pre_window:start]
            
            # Get sentence containing the entity
            entity_token = doc[start]
            entity_sent = None
            for sent in doc.sents:
                if sent.start <= start < sent.end:
                    entity_sent = sent
                    break
            
            # Check if negation is within the pre-context window and same sentence
            for token in pre_context:
                # Only consider negations within the same sentence
                if entity_sent and token.i >= entity_sent.start:
                    if token.text in negation_words or token.lemma_ in inverting_verbs:
                        # Check if this negation applies to our entity (using dependency parsing)
                        for dep_token in doc[start:end]:
                            # Check direct dependency
                            if dep_token.head == token or token.head == dep_token:
                                return True
                            
                            # Check indirect dependency (up to 2 levels)
                            if dep_token.head.head == token or token.head.head == dep_token:
                                return True
                            
                            # Check for negation of the predicate that relates to the entity
                            if (dep_token.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj'] and 
                                token.head == dep_token.head):
                                return True
                
                        # Check if the negation is applied to the verb governing the entity
                        if (start > 0 and doc[start-1].pos_ == "VERB" and 
                            (token.head == doc[start-1] or token.i == start-1)):
                            return True
            
            # Check for negation after the entity that might affect it
            post_window = min(len(doc), end + 5)
            post_context = doc[end:post_window]
            
            for token in post_context:
                # Only consider negations within the same sentence
                if entity_sent and token.i < entity_sent.end:
                    if token.text in negation_words:
                        # Check if this negation applies to our entity
                        if token.head.i >= start and token.head.i < end:
                            return True
            
            # If we've checked all possible ways and found no negation
            return False
        
        # If we don't know the entity position, do an improved general check
        has_negation = False
        
        # Check each sentence separately
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check for negation words in this sentence
            for token in sent:
                if token.text in negation_words or token.lemma_ in inverting_verbs:
                    has_negation = True
                    break
        
        return has_negation

    def adjust_sentiment_for_negation(self, sentiment_data, modifiers=None):
        """
        Adjust sentiment score based on sentiment modifiers including negation,
        intensifiers, and diminishers.
        
        Args:
            sentiment_data (dict): The sentiment data to adjust
            modifiers (dict, optional): Detected sentiment modifiers from detect_sentiment_modifiers
            
        Returns:
            dict: Adjusted sentiment data
        """
        # Create a copy to avoid modifying the original
        adjusted = sentiment_data.copy()
        
        # If no modifiers were provided, check if simple negation flag was passed
        if modifiers is None:
            # Default to no modifiers
            modifiers = {"has_negation": False, "sentiment_impact": 1.0}
        
        # Apply sentiment impact from modifiers
        impact = modifiers.get("sentiment_impact", 1.0)
        
        if modifiers.get("has_negation", False):
            # For negation, transform the score by flipping around the midpoint (0.5)
            # with a stronger effect that pushes further from neutral
            original_distance = adjusted["score"] - 0.5
            
            # Apply a non-linear transformation for stronger effect
            # Negative values will become more strongly positive and vice versa
            if impact < 0:
                # For negative impact (inverting verbs), apply stronger transformation
                # The further from neutral, the stronger the inversion effect
                adjusted["score"] = 0.5 - (original_distance * abs(impact) * (1.0 + abs(original_distance)))
            else:
                # For regular negation
                adjusted["score"] = 0.5 - (original_distance * impact)
            
            # Ensure the score stays within [0,1] range
            adjusted["score"] = max(0.0, min(1.0, adjusted["score"]))
        else:
            # For non-negations, apply the impact as a scaling factor from the neutral point
            original_distance = adjusted["score"] - 0.5
            adjusted["score"] = 0.5 + (original_distance * impact)
            
            # Ensure the score stays within [0,1] range
            adjusted["score"] = max(0.0, min(1.0, adjusted["score"]))
        
        # Update the label with more granular threshold
        if adjusted["score"] > 0.75:
            adjusted["label"] = "very positive"
        elif adjusted["score"] > 0.6:
            adjusted["label"] = "positive"
        elif adjusted["score"] > 0.45:
            adjusted["label"] = "slightly positive"
        elif adjusted["score"] > 0.35:
            adjusted["label"] = "neutral"
        elif adjusted["score"] > 0.2:
            adjusted["label"] = "slightly negative"
        elif adjusted["score"] > 0.05:
            adjusted["label"] = "negative"
        else:
            adjusted["label"] = "very negative"
        
        return adjusted
    def analyze_entity_sentiment(self, text):
        """
        Analyze sentiment for each entity in the text with enhanced context analysis.
        
        This method performs the following:
        1. Extracts entities from the text
        2. For each entity, finds relevant contexts
        3. Analyzes sentiment for each context, considering:
           - Context metadata (clause position, governing verbs)
           - Contrasting sentiments across different clauses
           - Sentiment modifiers (negation, intensifiers, diminishers)
        4. Combines context-level sentiments into entity-level sentiment
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            list: A list of entities with their associated sentiment information
        """
        try:
            # Get overall sentiment
            overall_sentiment = self.get_sentiment(text)
            
            # Get entities
            entities = self.get_entities(text)
            
            # Parse the text for dependency analysis
            doc = self.nlp(text)
            
            # Analyze sentiment for each entity
            results = []
            for entity, entity_type in entities:
                # Get weighted contexts containing the entity with enhanced metadata
                weighted_contexts = self.get_entity_contexts(text, entity)
                
                if weighted_contexts:
                    # Analyze sentiment for each context with weight consideration
                    context_sentiments = []
                    
                    # Group contexts by clause (to handle contrasting clauses)
                    clause_groups = {}
                    
                    for ctx, weight, ctx_meta in weighted_contexts:
                        # Get the sentiment for this specific context
                        context_sentiment = self.get_sentiment(ctx)
                        
                        # Find entity position in this context for targeted modifier detection
                        entity_pos = None
                        ctx_doc = self.nlp(ctx)
                        for i in range(len(ctx_doc) - len(entity.split()) + 1):
                            potential_entity = ' '.join([token.text for token in ctx_doc[i:i+len(entity.split())]]).lower()
                            if potential_entity == entity.lower():
                                entity_pos = (i, i+len(entity.split()))
                                break
                        
                        # Detect sentiment modifiers
                        if entity_pos:
                            modifiers = self.detect_sentiment_modifiers(ctx, entity_pos[0], entity_pos[1])
                        else:
                            # No specific position, check general modifiers
                            modifiers = self.detect_sentiment_modifiers(ctx)
                        
                        # Consider governing verb for sentiment adjustment
                        governing_verb = ctx_meta.get("governing_verb")
                        # Consider governing verb for sentiment adjustment
                        governing_verb = ctx_meta.get("governing_verb")
                        if governing_verb and governing_verb.lower() in self.inverting_verbs:
                            # If the governing verb is sentiment-inverting, apply stronger adjustment
                            modifiers["has_negation"] = True
                            # Strengthen the impact of inverting verbs (from -1.0 to -1.5)
                            modifiers["sentiment_impact"] *= -1.5
                        adjusted_sentiment = self.adjust_sentiment_for_negation(context_sentiment, modifiers)
                        
                        # Group by clause position for contrasting analysis
                        clause_pos = ctx_meta.get("clause_position", 0)
                        is_contrasting = ctx_meta.get("is_contrasting", False)
                        contains_entity = ctx_meta.get("contains_entity", True)
                        
                        # Adjust weight based on context metadata
                        adjusted_weight = weight
                        
                        # Boost weight if the context contains the entity significantly
                        if contains_entity:
                            adjusted_weight *= 2.0  # Increased from 1.5 to 2.0 for stronger entity presence
                        
                        # Boost weight for main clauses (usually first clause in English)
                        if clause_pos == 0 and not is_contrasting:
                            adjusted_weight *= 1.5  # Increased from 1.2 to 1.5 for main clause importance
                            
                        # Apply governing verb influence on weight
                        if governing_verb and governing_verb.lower() in self.inverting_verbs:
                            # If we have a strong sentiment verb directly governing our entity,
                            # it should have even more weight in determining sentiment
                            adjusted_weight *= 1.8
                        
                        # Store with adjusted weight and metadata
                        context_info = {
                            "sentiment": adjusted_sentiment,
                            "weight": adjusted_weight,
                            "meta": ctx_meta,
                            "text": ctx
                        }
                        
                        # Group by clause for contrasting analysis
                        key = f"{clause_pos}_{is_contrasting}"
                        if key not in clause_groups:
                            clause_groups[key] = []
                        clause_groups[key].append(context_info)
                        
                        # Also add to flat list for fallback analysis
                        context_sentiments.append((adjusted_sentiment, adjusted_weight))
                    
                    # Analyze contrasting clauses
                    contrasting_analysis = False
                    contrasting_details = []
                    
                    if len(clause_groups) > 1:
                        # Find clauses with the entity and their contrasting clauses
                        entity_clauses = []
                        contrast_clauses = []
                        
                        for key, contexts in clause_groups.items():
                            # Check if any context in this clause contains the entity
                            contains_entity = any(ctx["meta"].get("contains_entity", False) for ctx in contexts)
                            is_contrasting = any(ctx["meta"].get("is_contrasting", False) for ctx in contexts)
                            
                            if contains_entity:
                                entity_clauses.append((key, contexts))
                            if is_contrasting:
                                contrast_clauses.append((key, contexts))
                        
                        # If we have both entity clauses and contrasting clauses, analyze the contrast
                        if entity_clauses and contrast_clauses:
                            contrasting_analysis = True
                            
                            # Calculate average sentiment per clause with improved weighting
                            clause_sentiments = {}
                            clause_weights = {}
                            
                            for key, contexts in clause_groups.items():
                                # Sum all weights in this clause
                                total_weight = sum(ctx["weight"] for ctx in contexts)
                                clause_weights[key] = total_weight
                                
                                # Calculate the weighted sentiment score
                                weighted_score = sum(ctx["sentiment"]["score"] * ctx["weight"] for ctx in contexts) / total_weight
                                clause_sentiments[key] = weighted_score
                            
                            # Find the most contrasting clauses, prioritizing those with entities
                            max_contrast = 0
                            contrasting_pair = None
                            entity_clause_keys = [key for key, _ in entity_clauses]
                            
                            # First try to find contrast involving entity clauses
                            for key1 in entity_clause_keys:
                                for key2 in clause_sentiments:
                                    if key1 != key2:
                                        # Calculate contrast magnitude
                                        raw_contrast = abs(clause_sentiments[key1] - clause_sentiments[key2])
                                        
                                        # Weight the contrast by the reliability of the clauses
                                        # Higher weights mean more content and higher reliability
                                        reliability_factor = min(1.0, (clause_weights[key1] + clause_weights[key2]) / 10)
                                        effective_contrast = raw_contrast * reliability_factor
                                        
                                        if effective_contrast > max_contrast:
                                            max_contrast = effective_contrast
                                            contrasting_pair = (key1, key2)
                            
                            # If no good contrast with entity clauses, try any clauses
                            if max_contrast < 0.2:
                                for key1 in clause_sentiments:
                                    for key2 in clause_sentiments:
                                        if key1 != key2:
                                            raw_contrast = abs(clause_sentiments[key1] - clause_sentiments[key2])
                                            reliability_factor = min(1.0, (clause_weights[key1] + clause_weights[key2]) / 10)
                                            effective_contrast = raw_contrast * reliability_factor
                                            
                                            if effective_contrast > max_contrast:
                                                max_contrast = effective_contrast
                                                contrasting_pair = (key1, key2)
                            
                            # Lower the contrast threshold for detection
                            if contrasting_pair and max_contrast > 0.2:  # Reduced from 0.3 to 0.2
                                key1, key2 = contrasting_pair
                                
                                # Determine which clause contains the entity
                                key_with_entity = None
                                for key in [key1, key2]:
                                    contexts = clause_groups[key]
                                    if any(ctx["meta"].get("contains_entity", False) for ctx in contexts):
                                        key_with_entity = key
                                        break
                                
                                # If we found a clear clause with the entity, use its sentiment
                                # If we found a clear clause with the entity, use its sentiment
                                if key_with_entity:
                                    # Use the sentiment of the clause containing the entity with increased confidence
                                    contexts = clause_groups[key_with_entity]
                                    total_weight = sum(ctx["weight"] for ctx in contexts)
                                    
                                    # Calculate base weighted score
                                    base_weighted_score = sum(ctx["sentiment"]["score"] * ctx["weight"] for ctx in contexts) / total_weight
                                    
                                    # Check for sentiment-inverting governing verbs in this clause
                                    has_inverting_verb = any(
                                        ctx["meta"].get("governing_verb") and 
                                        ctx["meta"].get("governing_verb").lower() in self.inverting_verbs
                                        for ctx in contexts
                                    )
                                    
                                    # Adjust the sentiment more strongly if governing verbs are present
                                    if has_inverting_verb:
                                        # Push further from neutral (0.5) for stronger effect
                                        distance_from_neutral = base_weighted_score - 0.5
                                        weighted_score = 0.5 + (distance_from_neutral * 1.5)  # Amplify by 50%
                                    else:
                                        weighted_score = base_weighted_score
                                    
                                    # Record contrasting details for reporting
                                    contrasting_details.append({
                                        "primary_clause": key_with_entity,
                                        "contrasting_clause": key1 if key_with_entity == key2 else key2,
                                        "contrast_magnitude": max_contrast,
                                        "entity_sentiment": weighted_score
                                    })
                                    # weighted_score already set correctly above
                                    # No need for the redundant assignment
                                    weighted_score = weighted_score
                                else:
                                    # Fallback to standard weighted average
                                    total_weight = sum(weight for _, weight in context_sentiments)
                                    weighted_score = sum(s["score"] * weight for s, weight in context_sentiments) / total_weight
                            else:
                                # Not enough contrast, use standard weighted average
                                total_weight = sum(weight for _, weight in context_sentiments)
                                weighted_score = sum(s["score"] * weight for s, weight in context_sentiments) / total_weight
                        else:
                            # No clear contrasting structure, use standard weighted average
                            total_weight = sum(weight for _, weight in context_sentiments)
                            weighted_score = sum(s["score"] * weight for s, weight in context_sentiments) / total_weight
                    else:
                        # Only one clause group, use standard weighted average
                        total_weight = sum(weight for _, weight in context_sentiments)
                        weighted_score = sum(s["score"] * weight for s, weight in context_sentiments) / total_weight if total_weight > 0 else 0.5
                    
                    # Determine sentiment label based on weighted score
                    if weighted_score > 0.75:
                        sentiment_label = "very positive"
                    elif weighted_score > 0.6:
                        sentiment_label = "positive"
                    elif weighted_score > 0.45:
                        sentiment_label = "slightly positive"
                    elif weighted_score > 0.35:
                        sentiment_label = "neutral"
                    elif weighted_score > 0.2:
                        sentiment_label = "slightly negative"
                    elif weighted_score > 0.05:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "very negative"
                    
                    # Prepare the sentiment result
                    sentiment_result = {
                        "score": weighted_score,
                        "label": sentiment_label,
                    }
                    
                    # Prepare the entity result
                    entity_result = {
                        "entity": entity,
                        "entity_type": entity_type,
                        "sentiment": sentiment_result,
                        "contexts": [{
                            "text": ctx["text"],
                            "sentiment": ctx["sentiment"]["label"],
                            "weight": ctx["weight"],
                            "contains_entity": ctx["meta"]["contains_entity"],
                            "is_contrasting": ctx["meta"]["is_contrasting"]
                        } for ctx in [item for sublist in clause_groups.values() for item in sublist]]
                    }
                    
                    # Add contrasting details if found
                    if contrasting_analysis and contrasting_details:
                        entity_result["contrasting_details"] = contrasting_details
                    
                    results.append(entity_result)
                else:
                    # Fallback when no context is found
                    # Use global sentiment as a fallback with slightly reduced confidence
                    sentiment_result = overall_sentiment.copy()
                    
                    # Reduce confidence slightly as this is a fallback
                    sentiment_result["score"] = 0.5 + (sentiment_result["score"] - 0.5) * 0.8
                    
                    # Determine sentiment label based on adjusted score
                    if sentiment_result["score"] > 0.75:
                        sentiment_result["label"] = "very positive"
                    elif sentiment_result["score"] > 0.6:
                        sentiment_result["label"] = "positive"
                    elif sentiment_result["score"] > 0.45:
                        sentiment_result["label"] = "slightly positive"
                    elif sentiment_result["score"] > 0.35:
                        sentiment_result["label"] = "neutral"
                    elif sentiment_result["score"] > 0.2:
                        sentiment_result["label"] = "slightly negative"
                    elif sentiment_result["score"] > 0.05:
                        sentiment_result["label"] = "negative"
                    else:
                        sentiment_result["label"] = "very negative"
                    
                    # Create fallback result with warning
                    entity_result = {
                        "entity": entity,
                        "entity_type": entity_type,
                        "sentiment": sentiment_result,
                        "fallback": True,
                        "warning": "No specific context found for this entity, using adjusted global sentiment"
                    }
                    
                    results.append(entity_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in entity sentiment analysis: {e}")
            # Return error information instead of just raising to allow graceful degradation
            return [{
                "error": str(e),
                "entity_sentiment_analysis_failed": True,
                "overall_sentiment": self.get_sentiment(text) if text else {"score": 0.5, "label": "neutral"}
            }]
    
    def analyze_with_explanation(self, text):
        """
        Analyzes entity sentiment and provides a human-readable explanation.
        
        Args:
            text (str): The input text
            
        Returns:
            dict: Analysis results with explanations
        """
        # Get entity sentiment analysis
        entity_sentiments = self.analyze_entity_sentiment(text)
        
        # Get overall sentiment
        overall_sentiment = self.get_sentiment(text)
        
        # Generate explanations
        explanations = []
        for entity_data in entity_sentiments:
            entity = entity_data["entity"]
            sentiment = entity_data["sentiment"]["label"]
            score = entity_data["sentiment"]["score"]
            
            explanation = f"The entity '{entity}' has {sentiment} sentiment "
            explanation += f"(score: {score:.2f})."
            explanations.append(explanation)
        
        return {
            "text": text,
            "overall_sentiment": overall_sentiment,
            "entity_sentiments": entity_sentiments,
            "explanations": explanations
        }


def main():
    """
    Example usage of the EntitySentimentAnalyzer class.
    """
    # Create sample sentences
    samples = [
        "Chris hates Android phones but loves iPhones.",
        "Google released a new product that customers really enjoy.",
        "The restaurant has excellent food but terrible service.",
        "Microsoft Windows has some bugs, but Apple's MacOS isn't perfect either."
    ]
    
    # Initialize the analyzer
    try:
        analyzer = EntitySentimentAnalyzer()
        
        # Process each sample
        for sample in samples:
            print("\n" + "="*80)
            print(f"SAMPLE: {sample}")
            print("="*80)
            
            # Get detailed analysis with explanations
            results = analyzer.analyze_with_explanation(sample)
            
            # Print overall sentiment
            print(f"\nOverall sentiment: {results['overall_sentiment']['label']} "
                  f"(score: {results['overall_sentiment']['score']:.2f})")
            
            # Print entity sentiments
            print("\nEntity-level sentiment:")
            for entity in results["entity_sentiments"]:
                print(f"- {entity['entity']} ({entity['entity_type']}): "
                      f"{entity['sentiment']['label']} (score: {entity['sentiment']['score']:.2f})")
            
            # Print explanations
            print("\nExplanations:")
            for explanation in results["explanations"]:
                print(f"- {explanation}")
                
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

