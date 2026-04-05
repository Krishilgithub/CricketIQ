"""
src/chat/entity_extractor.py
────────────────────────────
Extracts necessary features (Teams, Venue, Toss) for the ML Prediction model.
"""

import requests
import json
from langsmith import traceable
from src.agents.sql_agent import OPENROUTER_API_KEY, OPENROUTER_URL, MODEL_ID

EXTRACTION_PROMPT = """You are an entity extraction engine for a Match Prediction Machine Learning Model.
Extract the following exact fields from the user's conversational request:

- "Team A": The primary team mentioned (MUST be extracted).
- "Team B": The opposing team (MUST be extracted, infer from history if necessary).
- "Venue": The stadium or city (OPTIONAL, output null if not specified).
- "Toss": The toss decision ("Bat" or "Field") (OPTIONAL, output null if not specified).

Output ONLY a JSON object. Do not wrap in markdown blocks. 
Example format:
{
    "Team A": "India",
    "Team B": "Australia",
    "Venue": "Wankhede Stadium",
    "Toss": "Bat"
}
"""

@traceable(run_type="chain", name="Prediction Entity Extraction")
def extract_prediction_entities(user_query: str, history: list) -> dict:
    """Extracts features needed for ML predictor."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8503",
        "X-Title": "CricketIQ Extract",
    }
    
    messages = [{"role": "system", "content": EXTRACTION_PROMPT}]
    
    for turn in history[-4:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"][:200]})
            
    messages.append({"role": "user", "content": user_query})
    
    # Default values needed by ML Model
    result = {
        "Team A": None,
        "Team B": None,
        "Venue": "Eden Gardens", # Default venue if none provided
        "Toss": "Bat"          # Default toss if none provided
    }
    
    try:
        response = requests.post(
            OPENROUTER_URL, 
            headers=headers, 
            json={"model": MODEL_ID, "messages": messages, "max_tokens": 150, "temperature": 0.0}, 
            timeout=15
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content)
        
        if data.get("Team A"): result["Team A"] = data["Team A"]
        if data.get("Team B"): result["Team B"] = data["Team B"]
        if data.get("Venue"): result["Venue"] = data["Venue"]
        if data.get("Toss"): result["Toss"] = data["Toss"]
        
        return result
    except Exception as e:
        print(f"Extraction Error: {e}")
        return result
