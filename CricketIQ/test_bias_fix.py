"""Detailed bias validation test for Phase 22."""
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["STREAMLIT_RUNTIME_MODE"] = "bare"

from src.ml.predictor import predict_match

tests = [
    ("India", "Australia", "Melbourne Cricket Ground", "Bat"),
    ("Australia", "India", "Melbourne Cricket Ground", "Bat"),  # Swapped
    ("Australia", "England", "Melbourne Cricket Ground", "Bat"),
    ("England", "Pakistan", "The Oval", "Bat"),
    ("India", "Pakistan", "Dubai International Cricket Stadium", "Bat"),
    ("New Zealand", "India", "Eden Gardens", "Field"),
    ("South Africa", "India", "Newlands", "Bat"),
]

print("=" * 80)
print("PHASE 22: DETAILED BIAS VALIDATION")
print("=" * 80)

india_favoured = 0
india_total = 0

for t1, t2, v, td in tests:
    r = predict_match(t1, t2, v, td)
    marker = ""
    if "India" in (t1, t2):
        india_total += 1
        if r["favourite"] == "India":
            india_favoured += 1
            marker = " [INDIA FAVOURED]"
        else:
            marker = " [INDIA NOT FAVOURED]"
    
    print(f"\n{t1} vs {t2} at {v}")
    print(f"  Toss: {td} | H2H: {r['h2h_rate']:.3f}")
    print(f"  {t1} Form: {r['team1_form']:.3f} | {t2} Form: {r['team2_form']:.3f}")
    print(f"  Prob({t1}): {r['win_prob_t1']:.3f}  Prob({t2}): {r['win_prob_t2']:.3f}")
    print(f"  >>> WINNER: {r['favourite']} ({r['fav_prob']:.1f}%, {r['confidence']}){marker}")

print("\n" + "=" * 80)
print(f"BIAS SUMMARY: India favoured in {india_favoured}/{india_total} matchups")

# Also check if probabilities are reasonable (not all > 60%)
print("\nPROBABILITY DISTRIBUTION:")
for t1, t2, v, td in tests:
    r = predict_match(t1, t2, v, td)
    spread = abs(r["win_prob_t1"] - 0.5)
    bar = "#" * int(spread * 100)
    print(f"  {t1:>15} vs {t2:<15} spread from 50/50: {spread:.3f} {bar}")

print("=" * 80)
