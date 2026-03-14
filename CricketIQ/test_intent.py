from dotenv import load_dotenv
load_dotenv()

from src.chat.intent_classifier import classify_intent
from src.chat.entity_extractor import extract_prediction_entities

q1 = "Will India beat Australia in the final?"
print(f"Query: {q1}")
intent = classify_intent(q1, [])
print(f"Intent: {intent}")
if intent == "PREDICTION":
    ent = extract_prediction_entities(q1, [])
    print(f"Entities: {ent}")

print("-" * 30)

q2 = "Who scored the most runs in 2022?"
print(f"Query: {q2}")
intent2 = classify_intent(q2, [])
print(f"Intent: {intent2}")
