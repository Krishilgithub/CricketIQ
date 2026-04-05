"""
src/chat/intent_classifier.py
─────────────────────────────
LLM-based Intent Classifier for routing chatbot queries.
"""

import requests
import json
from langsmith import traceable
from src.agents.sql_agent import OPENROUTER_API_KEY, OPENROUTER_URL, MODEL_ID

INTENT_PROMPT = """You are a highly capable intent classifier for a Cricket Analytics Chatbot.
Your job is to read the user's latest message, and classify the intent of their message into exactly ONE of the following categories:

1. "PREDICTION": The user is asking to predict the outcome of a future or hypothetical match, asking for win probabilities, chances of winning, or "who will win". IF the user asks "who will win", "who will win between", or "predict the match", YOU MUST CLASSIFY IT AS PREDICTION.
2. "SQL_ANALYTICS": The user is asking for historical data, statistics, past match results, player records.
3. "GENERAL_KNOWLEDGE": The user is just saying hello, asking general conversational questions, or asking something completely unrelated.

Output ONLY a JSON object with a single key "intent" mapping to one of the 3 categories above. Do not output markdown code blocks.
Example output: {"intent": "PREDICTION"}
"""

@traceable(run_type="chain", name="Intent Classification")
def classify_intent(user_query: str, history: list) -> str:
    """Classifies the user query intent to route to the correct pipeline."""
    # Hardcode a fast Regex fallback for obvious prediction queries to ensure 100% routing accuracy
    user_query_lower = user_query.lower()
    if "who will win" in user_query_lower or "predict" in user_query_lower or "chances of winning" in user_query_lower:
        return "PREDICTION"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8503",
        "X-Title": "CricketIQ Intent",
    }
    
    messages = [{"role": "system", "content": INTENT_PROMPT}]
    
    # Add a bit of context if available
    for turn in history[-2:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"][:200]})
    
    messages.append({"role": "user", "content": user_query})
    
    try:
        response = requests.post(
            OPENROUTER_URL, 
            headers=headers, 
            json={"model": MODEL_ID, "messages": messages, "max_tokens": 20, "temperature": 0.0}, 
            timeout=10
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Clean up any potential markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content)
        intent = data.get("intent", "SQL_ANALYTICS") # Default to SQL if unsure
        if intent not in ["PREDICTION", "SQL_ANALYTICS", "GENERAL_KNOWLEDGE"]:
            intent = "SQL_ANALYTICS"
        return intent
    except Exception as e:
        print(f"Intent Classification Error: {e}")
        return "SQL_ANALYTICS" # Fallback
