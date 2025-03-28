"""
Test data for entity-level sentiment analysis tests.
"""

# Copyright 2025 Chris Collins <chris@hitorro.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
TEST_TEXTS = {
    # Basic examples with clear entity sentiment
    "positive": "Apple has created amazing products that customers love.",
    "negative": "Samsung recalled their Galaxy Note 7 after battery explosions.",
    "mixed": "Microsoft has great software engineers but their products have bugs.",
    
    # Examples with negations
    "negation": "The iPhone isn't as good as reviewers claim.",
    "double_negation": "Sony doesn't make products that don't impress people.",
    
    # Examples with contrasting sentiments
    "contrast_simple": "Google has excellent search but poor privacy practices.",
    "contrast_complex": "While Facebook provides good connectivity with friends, it terrible for privacy and mental health.",
    
    # Examples with sentiment-inverting verbs
    "inverting_verbs": "Many consumers hate Comcast's customer service but love their internet speed.",
    
    # Examples with intensifiers and diminishers
    "modifiers": "Tesla makes extremely innovative cars but slightly overpriced vehicles.",
    
    # Examples with multiple entities
    "multiple_entities": "Apple and Google compete in the smartphone market, while Microsoft focuses more on enterprise software.",
    
    # Complex sentences
    "complex_sentence": "Amazon's CEO, who recently stepped down, built a company that dominates e-commerce but struggles with worker conditions.",
    
    # Multi-paragraph text
    "multi_paragraph": """
    The tech industry has seen major changes in recent years. Companies like Google and Facebook have dominated the advertising market, while Apple and Samsung compete in the smartphone space.
    
    Microsoft has pivoted to cloud services, making Azure a strong competitor to Amazon's AWS. However, their consumer products have had mixed success with some hits like Xbox and misses like Windows Phone.
    
    Tesla has revolutionized the automotive industry with their electric vehicles. Despite production challenges, they've maintained strong customer loyalty. Their autonomous driving features are ahead of competitors but also face regulatory scrutiny.
    """,
    
    # Edge cases
    "no_entities": "The product works well when used properly.",
    "repeated_entities": "Amazon ships Amazon products from Amazon warehouses using Amazon logistics.",
    
    # Industry-specific examples
    "tech_industry": "Intel's new processors are faster but less energy-efficient than AMD's latest offerings.",
    "retail_industry": "Walmart offers lower prices than Target but Target provides a better shopping experience.",
    "automotive_industry": "Ford's F-150 Lightning electric truck demonstrates strong innovation but faces production delays.",
    
    # Examples with subtle sentiment
    "subtle_positive": "The Netflix interface has been refined over several iterations.",
    "subtle_negative": "Uber's algorithm sometimes takes longer routes than necessary.",
    
    # Examples with mixed entity types
    "mixed_entities": "While Tim Cook leads Apple effectively, the company's hardware repair policies frustrate many users."
}

# Test cases for specific features
NEGATION_TEST_CASES = [
    "Apple doesn't make bad products.",
    "Google isn't failing in the search market.",
    "Amazon doesn't ship orders slowly.",
    "Microsoft hasn't ignored security issues.",
    "Facebook isn't protecting user privacy adequately."
]

CONTRASTING_TEST_CASES = [
    "Twitter has an active user base but struggles with content moderation.",
    "Netflix produces great original content but their library of older movies is shrinking.",
    "Amazon delivers quickly but sometimes packages arrive damaged.",
    "Spotify offers a vast music collection but pays artists poorly.",
    "Samsung phones have excellent hardware but too much bloatware."
]

INVERTING_VERBS_TEST_CASES = [
    "Customers love Costco's return policy.",
    "Critics hate EA's microtransaction strategy.",
    "Users adore Reddit's community features.",
    "Environmentalists despise Shell's drilling practices.",
    "Gamers enjoy Nintendo's first-party titles."
]

# Test data for API testing
API_TEST_CASES = {
    "single_sentence": "Apple creates innovative products.",
    "multi_sentence": "Microsoft has good cloud services. Their Office suite is popular. However, Windows has some issues.",
    "long_text": " ".join(["The technology landscape is constantly evolving."] * 50),
    "empty": "",
    "special_chars": "Google's AI & machine learning capabilities are industry-leading!",
}

# Edge cases and error conditions
ERROR_TEST_CASES = {
    "very

