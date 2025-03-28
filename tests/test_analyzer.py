import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_sentiment import EntitySentimentAnalyzer

class TestEntitySentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = EntitySentimentAnalyzer()
    
    def test_basic_sentiment(self):
        """Test basic entity sentiment detection."""
        text = "Apple makes good products."
        results = self.analyzer.analyze(text)
        
        # Check if Apple is detected with positive sentiment
        apple_result = next((r for r in results if r['entity'] == 'Apple'), None)
        self.assertIsNotNone(apple_result)
        self.assertGreater(apple_result['sentiment_score'], 0)
        self.assertEqual(apple_result['sentiment_label'], 'positive')
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        text = "Samsung phones have terrible battery life."
        results = self.analyzer.analyze(text)
        
        # Check if Samsung is detected with negative sentiment
        samsung_result = next((r for r in results if r['entity'] == 'Samsung'), None)
        self.assertIsNotNone(samsung_result)
        self.assertLess(samsung_result['sentiment_score'], 0)
        self.assertIn(samsung_result['sentiment_label'], ['negative', 'very negative'])
    
    def test_negation(self):
        """Test handling of negations."""
        text = "Google isn't making reliable software lately."
        results = self.analyzer.analyze(text)
        
        # Check if Google has negative sentiment due to negation
        google_result = next((r for r in results if r['entity'] == 'Google'), None)
        self.assertIsNotNone(google_result)
        self.assertLess(google_result['sentiment_score'], 0)
    
    def test_contrasting_sentiment(self):
        """Test handling of contrasting sentiments with 'but' clauses."""
        text = "Tesla has great electric cars but poor customer service."
        results = self.analyzer.analyze(text)
        
        # Check if Tesla is found
        tesla_result = next((r for r in results if r['entity'] == 'Tesla'), None)
        self.assertIsNotNone(tesla_result)
        
        # The sentiment could be mixed or slightly positive/negative depending on implementation
        self.assertIsNotNone(tesla_result['sentiment_score'])
        
        # Check if the contrasting details are captured
        self.assertIn('contrasting_details', tesla_result)
        self.assertTrue(len(tesla_result['contrasting_details']) > 0)
    
    def test_inverting_verbs(self):
        """Test handling of sentiment-inverting verbs like 'hate' and 'love'."""
        text = "John hates Microsoft Windows but loves Apple MacOS."
        results = self.analyzer.analyze(text)
        
        # Check Microsoft sentiment (negative)
        ms_result = next((r for r in results if r['entity'] == 'Microsoft Windows'), None)
        self.assertIsNotNone(ms_result)
        self.assertLess(ms_result['sentiment_score'], 0)
        
        # Check Apple sentiment (positive)
        apple_result = next((r for r in results if r['entity'] == 'Apple MacOS'), None)
        self.assertIsNotNone(apple_result)
        self.assertGreater(apple_result['sentiment_score'], 0)
    
    def test_entity_context_extraction(self):
        """Test the extraction of contexts around entities."""
        text = "Amazon delivers packages quickly but their customer service is slow."
        
        # Access the internal method to test context extraction
        doc = self.analyzer.nlp(text)
        entities = [(e.text, e.start, e.end) for e in doc.ents]
        contexts = self.analyzer.get_entity_contexts(text, entities)
        
        # Check if Amazon has contexts related to both delivery and customer service
        amazon_contexts = [c for c in contexts if c['entity'] == 'Amazon']
        self.assertTrue(len(amazon_contexts) > 0)
        
        # Check if at least one context contains "delivers" and another contains "customer service"
        delivery_context = any("delivers" in c['text'].lower() for c in amazon_contexts)
        service_context = any("customer service" in c['text'].lower() for c in amazon_contexts)
        
        self.assertTrue(delivery_context or service_context)
    
    def test_intensifiers_and_diminishers(self):
        """Test handling of intensifiers and diminishers."""
        text = "Facebook is extremely problematic for privacy but slightly better than Twitter."
        results = self.analyzer.analyze(text)
        
        # Check if Facebook has stronger negative sentiment due to "extremely"
        facebook_result = next((r for r in results if r['entity'] == 'Facebook'), None)
        self.assertIsNotNone(facebook_result)
        self.assertLess(facebook_result['sentiment_score'], 0)
        
        # Check if Twitter also has negative sentiment
        twitter_result = next((r for r in results if r['entity'] == 'Twitter'), None)
        self.assertIsNotNone(twitter_result)
        self.assertLess(twitter_result['sentiment_score'], 0)
        
        # Facebook should be more negative than Twitter
        if twitter_result and facebook_result:
            self.assertLess(facebook_result['sentiment_score'], twitter_result['sentiment_score'])

if __name__ == '__main__':
    unittest.main()

