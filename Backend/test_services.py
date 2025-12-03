"""Test script for NLP and Classification services."""

from classification_service import get_classification_service
from nlp_service import get_nlp_service

print("=" * 60)
print("FIR System - Service Tests")
print("=" * 60)

# Test classification
print("\n--- Classification Tests ---")
cls = get_classification_service()
test_cases = [
    "My mobile phone was stolen at MG Road yesterday",
    "Someone attacked me and beat me badly",
    "I received a fake OTP call and lost money from bank account",
    "He cheated me and took Rs 50000",
    "My neighbor has been harassing me daily",
]

for text in test_cases:
    result = cls.classify(text)
    print(f"'{text[:45]}...' -> {result.label} ({result.confidence:.1%})")

# Test entity extraction
print("\n--- Entity Extraction Tests ---")
nlp = get_nlp_service()
text = "My mobile phone was stolen at MG Road, Bangalore on 15th November 2024 around 8 PM. The value was Rs 50000."
entities = nlp.extract_entities(text)
print(f"Text: {text}")
print(f"Locations: {entities['locations']}")
print(f"Dates: {entities['dates']}")
print(f"Times: {entities['times']}")
print(f"Money: {entities['money']}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
