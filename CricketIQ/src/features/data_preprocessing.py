"""
src/features/data_preprocessing.py
──────────────────────────────────
Builds scikit-learn preprocessing pipelines for numerical features.
Phase 22 FIX: Removed team_1, team_2, venue from categorical features
to prevent team-identity leakage. The model should learn from
strength/form/h2h signals, NOT from team names.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Phase 22: ALL features are numerical now.
# Team identity features REMOVED to prevent bias.
NUMERICAL_FEATURES = [
    "toss_bat",
    "venue_avg_1st_inns_runs", 
    "venue_chase_success_rate",
    "h2h_advantage",
    "form_last5_diff",
    "form_last10_diff",
    "momentum_diff",
    "venue_win_rate_diff",
]

# Phase 22: No categorical features — eliminates OneHotEncoding team identity
CATEGORICAL_FEATURES = []

TARGET_COL = "team_1_win"

def get_preprocessor():
    """
    Returns a compiled Scikit-Learn ColumnTransformer.
    Phase 22: Numerical-only pipeline (no OneHotEncoding of team names).
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
        ])
        
    return preprocessor

def get_training_columns():
    return NUMERICAL_FEATURES + CATEGORICAL_FEATURES, TARGET_COL
