import json
import uuid
import time
import os
import random
from pathlib import Path
from datetime import datetime

DROP_FOLDER = Path("data/raw/new_json_drops")

def generate_mock_live_match():
    """Generates a mock Cricsheet JSON file for real-time testing."""
    DROP_FOLDER.mkdir(parents=True, exist_ok=True)
    
    match_id = random.randint(3000000, 4000000)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    teams = ["India", "Australia"]
    toss_winner = random.choice(teams)
    
    data = {
        "meta": {
            "data_version": "1.1.0",
            "created": current_date,
            "revision": 1
        },
        "info": {
            "dates": [current_date],
            "event": {"name": "ICC T20 World Cup 2026", "match_number": 56},
            "match_type": "T20",
            "gender": "male",
            "city": "Mumbai",
            "venue": "Wankhede Stadium",
            "teams": teams,
            "toss": {"winner": toss_winner, "decision": "bat"},
            "outcome": {"winner": toss_winner, "by": {"runs": random.randint(10, 50)}},
            "players": {
                teams[0]: ["Player A", "Player B", "Player C"],
                teams[1]: ["Player X", "Player Y", "Player Z"]
            }
        },
        "innings": [
            {
                "team": toss_winner,
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Player A",
                                "bowler": "Player X",
                                "non_striker": "Player B",
                                "runs": {"batter": 4, "extras": 0, "total": 4}
                            },
                            {
                                "batter": "Player A",
                                "bowler": "Player X",
                                "non_striker": "Player B",
                                "runs": {"batter": 6, "extras": 0, "total": 6}
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    file_path = DROP_FOLDER / f"{match_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Generated Mock Live Match JSON: {file_path}")

if __name__ == "__main__":
    generate_mock_live_match()
