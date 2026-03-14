"""
src/features/data_preprocessing.py
──────────────────────────────────
Builds scikit-learn preprocessing pipelines for numerical and categorical features.
Used by the model training module to standardize imputing, scaling, and encoding.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define robust categorizations
NUMERICAL_FEATURES = [
    "venue_avg_1st_inns_runs", 
    "venue_chase_success_rate",
    "team_1_h2h_win_rate", 
    "team_1_form_last5", 
    "team_1_form_last10", 
    "team_1_momentum",
    "team_2_form_last5", 
    "team_2_form_last10", 
    "team_2_momentum",
    "team_1_venue_win_rate", 
    "team_2_venue_win_rate"
]

CATEGORICAL_FEATURES = [
    "toss_bat",
    "toss_winner_is_team_1",
    "venue",
    "team_1",
    "team_2"
]

TARGET_COL = "team_1_win"

def get_preprocessor():
    """
    Returns a compiled Scikit-Learn ColumnTransformer.
    - Numerical: Impute median -> StandardScale
    - Categorical: Impute mode -> OneHotEncode (handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
        
    return preprocessor

def get_training_columns():
    return NUMERICAL_FEATURES + CATEGORICAL_FEATURES, TARGET_COL
