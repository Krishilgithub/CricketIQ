"""
src/ingestion/canonical_mappings.py
─────────────────────────────────────
Canonical team name and venue name mappings.

Cricsheet uses slightly different team/venue name variants across seasons
(e.g. "U.A.E." vs "United Arab Emirates", "Dubai International Cricket Stadium"
vs "Dubai (DSC)"). This module provides:

 1. TEAM_ALIASES   – maps any known alias → canonical team name
 2. VENUE_ALIASES  – maps any known alias → canonical venue name
 3. `standardize_team(name)`  – returns canonical or original if unmapped
 4. `standardize_venue(name)` – returns canonical or original if unmapped
 5. `apply_canonical_teams(df, cols)` – applies to a DataFrame in-place
"""

from __future__ import annotations

import pandas as pd

# ── Team name aliases ─────────────────────────────────────────────────────────
# Map every known alias → canonical ICC team name
TEAM_ALIASES: dict[str, str] = {
    # United Arab Emirates
    "U.A.E.": "United Arab Emirates",
    "UAE": "United Arab Emirates",
    # United States
    "U.S.A.": "United States of America",
    "USA": "United States of America",
    "US": "United States of America",
    # Papua New Guinea
    "P.N.G.": "Papua New Guinea",
    "PNG": "Papua New Guinea",
    # West Indies
    "West Indies": "West Indies",
    "WI": "West Indies",
    # Netherlands
    "Netherlands": "Netherlands",
    "Holland": "Netherlands",
    # Sri Lanka
    "Sri Lanka": "Sri Lanka",
    "SL": "Sri Lanka",
    # New Zealand
    "New Zealand": "New Zealand",
    "NZ": "New Zealand",
    # South Africa
    "South Africa": "South Africa",
    "SA": "South Africa",
    # Pakistan
    "Pakistan": "Pakistan",
    "PAK": "Pakistan",
    # India
    "India": "India",
    "IND": "India",
    # Australia
    "Australia": "Australia",
    "AUS": "Australia",
    # England
    "England": "England",
    "ENG": "England",
    # Bangladesh
    "Bangladesh": "Bangladesh",
    "BAN": "Bangladesh",
    # Zimbabwe
    "Zimbabwe": "Zimbabwe",
    "ZIM": "Zimbabwe",
    # Afghanistan
    "Afghanistan": "Afghanistan",
    "AFG": "Afghanistan",
    # Scotland
    "Scotland": "Scotland",
    "SCO": "Scotland",
    # Ireland
    "Ireland": "Ireland",
    "IRE": "Ireland",
    # Namibia
    "Namibia": "Namibia",
    "NAM": "Namibia",
    # Oman
    "Oman": "Oman",
    # Nepal
    "Nepal": "Nepal",
    # Canada
    "Canada": "Canada",
    # Hong Kong
    "Hong Kong": "Hong Kong",
    # Jersey
    "Jersey": "Jersey",
    # Kenya
    "Kenya": "Kenya",
}

# ── Venue name aliases ────────────────────────────────────────────────────────
# Map every known alias → canonical venue name
VENUE_ALIASES: dict[str, str] = {
    # India
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Wankhede Stadium, Bombay": "Wankhede Stadium",
    "M. Chinnaswamy Stadium, Bangalore": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Rajiv Gandhi International Cricket Stadium": "Rajiv Gandhi International Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium, Mohali",
    "PCA Stadium": "PCA Stadium, Mohali",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    # Australia
    "Melbourne Cricket Ground": "Melbourne Cricket Ground",
    "MCG": "Melbourne Cricket Ground",
    "Sydney Cricket Ground": "Sydney Cricket Ground",
    "SCG": "Sydney Cricket Ground",
    "Perth Stadium": "Perth Stadium",
    "Optus Stadium": "Perth Stadium",
    "Adelaide Oval": "Adelaide Oval",
    "Brisbane Cricket Ground": "The Gabba",
    "The Gabba, Brisbane": "The Gabba",
    # England
    "Lord's Cricket Ground": "Lord's",
    "Lords": "Lord's",
    "The Oval": "The Kia Oval",
    "Kia Oval": "The Kia Oval",
    "Old Trafford": "Emirates Old Trafford",
    # South Africa
    "Newlands Cricket Ground": "Newlands",
    "SuperSport Park": "SuperSport Park",
    "Centurion": "SuperSport Park",
    # West Indies
    "Kensington Oval, Bridgetown": "Kensington Oval",
    "Providence Stadium": "Providence Stadium",
    # UAE
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
    "Dubai (DSC)": "Dubai International Cricket Stadium",
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium",
    "Abu Dhabi": "Sheikh Zayed Stadium",
}


def standardize_team(name: str | None) -> str | None:
    """Return the canonical team name, or the original if no mapping exists."""
    if name is None or (isinstance(name, float)):
        return name
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def standardize_venue(name: str | None) -> str | None:
    """Return the canonical venue name, or the original if no mapping exists."""
    if name is None or (isinstance(name, float)):
        return name
    return VENUE_ALIASES.get(str(name).strip(), str(name).strip())


def apply_canonical_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply team name standardization to specified columns of a DataFrame.
    Modifies the DataFrame in-place and returns it.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: standardize_team(x) if isinstance(x, str) else x
            )
    return df


def apply_canonical_venues(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply venue name standardization to specified columns of a DataFrame.
    Modifies the DataFrame in-place and returns it.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: standardize_venue(x) if isinstance(x, str) else x
            )
    return df
