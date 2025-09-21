#!/usr/bin/env python3
"""
Plan A fetch: PSCompPars from NASA Exoplanet Archive TAP → local JSON + Parquet + DuckDB

Requirements:
  pip install requests pandas pyarrow duckdb

Run:
  python fetch_exoplanets.py
"""

import os, json, datetime, pathlib, urllib.parse
import requests
import pandas as pd
import duckdb

# 1. Config
BASE = pathlib.Path(__file__).resolve().parent
DATA_RAW = BASE / "data" / "raw"
DATA_NORM = BASE / "data" / "normalized"
DB_PATH = BASE / "data" / "exoplanets.duckdb"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_NORM.mkdir(parents=True, exist_ok=True)

# Choose table: pscomppars is unified and recommended. Use k2pandc if you want only K2.
TABLE = "pscomppars"

# Columns we need for the score and basic UI
COLUMNS = [
    "pl_name","hostname","pl_insol","pl_rade","pl_eqt",
    "pl_bmasse","pl_bmassj","pl_orbeccen","pl_orbper",
    "st_teff","st_rad","st_mass",
    "pl_orbsmax","pl_dens","pl_trandep",
    "sy_dist","disc_year"
]

# Filter to the default solution rows - not needed for pscomppars
WHERE = ""

# Build TAP query
if WHERE:
    sql = f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE {WHERE}"
else:
    sql = f"SELECT {', '.join(COLUMNS)} FROM {TABLE}"
encoded_sql = urllib.parse.quote_plus(sql)

# TAP sync endpoint
URL = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
PARAMS = {
    "query": sql,          # requests will URL encode properly
    "format": "json",
    "MAXREC": 500000       # high ceiling to avoid truncation
}

def fetch_tap_json(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()

def to_dataframe(tap_json: dict) -> pd.DataFrame:
    # TAP json is usually a list of dict rows
    # but sometimes returns {"data":[...]} shape in other services.
    if isinstance(tap_json, dict) and "data" in tap_json:
        data = tap_json["data"]
    else:
        data = tap_json
    df = pd.DataFrame(data)
    # Ensure all selected columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Type clean numerics where appropriate
    numeric_cols = [
        "pl_insol","pl_rade","pl_eqt","pl_bmasse","pl_bmassj",
        "pl_orbeccen","pl_orbper","st_teff","st_rad","st_mass",
        "pl_orbsmax","pl_dens","pl_trandep","sy_dist"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["disc_year"] = pd.to_numeric(df["disc_year"], errors="coerce").astype("Int64")
    return df

def main():
    fetch_date = datetime.date.today().strftime("%Y%m%d")
    raw_path = DATA_RAW / f"{TABLE}_{fetch_date}.json"
    parquet_path = DATA_NORM / f"{TABLE}_{fetch_date}.parquet"

    print("Fetching from TAP…")
    tap_json = fetch_tap_json(URL, PARAMS)
    print(f"Rows received: {len(tap_json) if isinstance(tap_json, list) else len(tap_json.get('data', []))}")

    print(f"Saving raw JSON → {raw_path}")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(tap_json, f, ensure_ascii=False)

    print("Normalizing to DataFrame…")
    df = to_dataframe(tap_json)

    # Require pl_insol and pl_rade for our axioms
    before = len(df)
    df = df.dropna(subset=["pl_insol","pl_rade"])
    print(f"Kept {len(df)}/{before} rows with both pl_insol and pl_rade")

    print(f"Writing Parquet → {parquet_path}")
    df.to_parquet(parquet_path, index=False)

    # Create or update DuckDB and load a raw_ps table
    print(f"Updating DuckDB at {DB_PATH}")
    con = duckdb.connect(DB_PATH)
    # Create schema if not exists
    con.execute("CREATE SCHEMA IF NOT EXISTS exo")
    # Load parquet into a permanent table exo.raw_ps
    con.execute("DROP TABLE IF EXISTS exo.raw_ps")
    con.execute(f"CREATE TABLE exo.raw_ps AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')")
    # Helpful indexes for quick lookup
    con.execute("CREATE INDEX IF NOT EXISTS idx_raw_ps_plname ON exo.raw_ps(pl_name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_raw_ps_host ON exo.raw_ps(hostname)")
    con.close()

    print("Done. Files created:")
    print(f"  {raw_path}")
    print(f"  {parquet_path}")
    print(f"  {DB_PATH} (schema exo, table raw_ps)")

if __name__ == "__main__":
    main()
