from dotenv import load_dotenv
load_dotenv()

from src.agents.sql_agent import agent_loop
from src.ml.predictor import predict_match

print("--- Testing ML Prediction Trace ---")
try:
    pred = predict_match("India", "Australia", "Wankhede Stadium", "Bat")
    print(f"Prediction success! {pred['team1']} Win Prob: {pred['win_prob_t1']:.2f}")
except Exception as e:
    print("ML error:", e)

print("\n--- Testing SQL Agent Trace ---")
try:
    reply = agent_loop("Who won the 2007 T20 World Cup?", history=[])
    print("Agent Reply:", reply[:150] + "...")
except Exception as e:
    print("Agent error:", e)
