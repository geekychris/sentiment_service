import unittest
import json
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import app
from tests.test_data import TEST_TEXTS

class TestSentimentAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
    
    def test_analyze_single_sentence(self):
        """Test analyzing a single sentence."""
        data = {
            "text": "Apple makes innovative products."
        }
        response = self.client.post("/analyze", json=data)
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)
        
        # Check if we have sentences and entities
        self.assertIn("sentences", result)
        self.assertGreater(len(result["sentences"]), 0)
        
        # Check for Apple entity
        apple_found = False
        for sentence_result in result["results"]:
            for entity_result in sentence_result["entities"]:
                if entity_result["entity"] == "Apple":
                    apple_found = True
                    self.assertIn("sentiment_score", entity_result)
                    self.assertIn("sentiment_label", entity_result)
        
        self.assertTrue(apple_found, "Apple entity not found in results")
    
    def test_analyze_multi_paragraph(self):
        """Test analyzing multi-paragraph text."""
        data = {
            "text": TEST_TEXTS["multi_paragraph"]
        }
        response = self.client.post("/analyze", json=data)
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn("results", result)
        
        # Should have multiple sentences
        self.assertIn("sentences", result)
        self.assertGreater(len(result["sentences"]), 3)
        
        # Check if entities were found across paragraphs
        entity_count = sum(len(sent["entities"]) for sent in result["results"])
        self.assertGreater(entity_count, 2)
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        # Empty text
        data = {
            "text": ""
        }
        response = self.client.post("/analyze", json=data)
        self.assertEqual(response.status_code, 400)
        
        # Very long text (should be handled gracefully)
        data = {
            "text": "This is a test. " * 1000
        }
        response = self.client.post("/analyze", json=data)
        self.assertEqual(response.status_code, 200)
    
    def test_contrasting_sentiment_api(self):
        """Test the API with text containing contrasting sentiments."""
        data = {
            "text": "Microsoft is a great company but their Windows OS has some bugs."
        }
        response = self.client.post("/analyze", json=data)
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        
        # Find Microsoft entity
        microsoft_found = False
        for sentence_result in result["results"]:
            for entity_result in sentence_result["entities"]:
                if entity_result["entity"] == "Microsoft":
                    microsoft_found = True
                    self.assertIn("contrasting_details", entity_result)
        
        self.assertTrue(microsoft_found, "Microsoft entity not found in results")
    
    def test_batch_analysis(self):
        """Test batch analysis of multiple texts."""
        data = {
            "texts": [
                "Google has excellent search technology.",
                "Amazon's delivery service is unreliable sometimes."
            ]
        }
        response = self.client.post("/analyze/batch", json=data)
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn("batch_results", result)
        self.assertEqual(len(result["batch_results"]), 2)
        
        # Check first text results
        first_result = result["batch_results"][0]
        self.assertIn("results", first_result)
        
        # Check second text results
        second_result = result["batch_results"][1]
        self.assertIn("results", second_result)
        
        # Entities should be found in each text
        google_found = False
        amazon_found = False
        
        for sentence_result in first_result["results"]:
            for entity_result in sentence_result["entities"]:
                if entity_result["entity"] == "Google":
                    google_found = True
        
        for sentence_result in second_result["results"]:
            for entity_result in sentence_result["entities"]:
                if entity_result["entity"] == "Amazon":
                    amazon_found = True
        
        self.assertTrue(google_found, "Google entity not found in first text results")
        self.assertTrue(amazon_found, "Amazon entity not found in second text results")

if __name__ == '__main__':
    unittest.main()

