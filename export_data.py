#!/usr/bin/env python3
"""
export_data.py - Export DuckDB data to various formats for web deployment

Note: This script was primarily created using AI assistance.

Exports exoplanet data from DuckDB to:
- JSON files (for static hosting)
- CSV files (for data sharing)
- Parquet files (for efficient storage)
- SQLite database (for portable deployment)
"""

import os
import json
import sqlite3
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

# Configuration
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "exoplanets.duckdb"
EXPORT_DIR = BASE / "exports"

def clean_data_for_json(data):
    """Clean data to make it JSON serializable"""
    if isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        if pd.isna(data) or np.isnan(data):
            return None
        return float(data) if isinstance(data, np.floating) else int(data)
    elif pd.isna(data):
        return None
    else:
        return data

def export_to_json():
    """Export data to JSON files"""
    print("📄 Exporting to JSON...")
    
    con = duckdb.connect(str(DB_PATH))
    
    # Export full dataset
    df = con.execute("SELECT * FROM exo.raw_ps").df()
    data = df.to_dict('records')
    cleaned_data = clean_data_for_json(data)
    
    json_dir = EXPORT_DIR / "json"
    json_dir.mkdir(exist_ok=True)
    
    with open(json_dir / "exoplanets_full.json", 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Export smaller subsets for faster loading
    # Top 1000 by habitability
    top_1000 = cleaned_data[:1000]
    with open(json_dir / "exoplanets_top1000.json", 'w') as f:
        json.dump(top_1000, f, indent=2)
    
    # Export metadata
    stats = {
        "total_planets": len(cleaned_data),
        "export_timestamp": pd.Timestamp.now().isoformat(),
        "fields": list(df.columns) if len(df) > 0 else []
    }
    
    with open(json_dir / "metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    con.close()
    print(f"✅ JSON export complete: {len(cleaned_data)} planets")

def export_to_csv():
    """Export data to CSV files"""
    print("📊 Exporting to CSV...")
    
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM exo.raw_ps").df()
    
    csv_dir = EXPORT_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)
    
    # Full dataset
    df.to_csv(csv_dir / "exoplanets_full.csv", index=False)
    
    # Essential columns only (smaller file)
    essential_cols = [
        'pl_name', 'hostname', 'pl_rade', 'pl_bmasse', 'pl_orbper', 
        'pl_insol', 'pl_eqt', 'sy_dist', 'disc_year', 'description'
    ]
    
    available_cols = [col for col in essential_cols if col in df.columns]
    df_essential = df[available_cols]
    df_essential.to_csv(csv_dir / "exoplanets_essential.csv", index=False)
    
    con.close()
    print(f"✅ CSV export complete: {len(df)} planets")

def export_to_parquet():
    """Export data to Parquet files"""
    print("🗜️ Exporting to Parquet...")
    
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM exo.raw_ps").df()
    
    parquet_dir = EXPORT_DIR / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    
    # Full dataset
    df.to_parquet(parquet_dir / "exoplanets_full.parquet", index=False)
    
    con.close()
    print(f"✅ Parquet export complete: {len(df)} planets")

def export_to_sqlite():
    """Export data to SQLite database"""
    print("🗃️ Exporting to SQLite...")
    
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM exo.raw_ps").df()
    
    sqlite_dir = EXPORT_DIR / "sqlite"
    sqlite_dir.mkdir(exist_ok=True)
    
    sqlite_path = sqlite_dir / "exoplanets.sqlite"
    
    # Create SQLite connection and export
    sqlite_con = sqlite3.connect(sqlite_path)
    df.to_sql('exoplanets', sqlite_con, if_exists='replace', index=False)
    sqlite_con.close()
    
    con.close()
    print(f"✅ SQLite export complete: {len(df)} planets")

def create_static_api_files():
    """Create static JSON files that mimic API responses"""
    print("🌐 Creating static API files...")
    
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM exo.raw_ps").df()
    
    api_dir = EXPORT_DIR / "api"
    api_dir.mkdir(exist_ok=True)
    
    # Simulate the app.py habitability scoring
    data = df.to_dict('records')
    
    # Add habitability scores (simplified version)
    for planet in data:
        score = 0
        if planet.get('pl_insol') and not pd.isna(planet['pl_insol']):
            insolation = planet['pl_insol']
            if 0.3 <= insolation <= 1.5:
                score += min(40, 40 * insolation) if insolation <= 1.0 else min(40, 40 * (1.5 - insolation) / 0.5)
        
        planet['habitability_score'] = round(score, 1)
    
    # Sort by habitability
    data.sort(key=lambda x: x.get('habitability_score', 0), reverse=True)
    
    cleaned_data = clean_data_for_json(data)
    
    # Create API-like responses
    api_response = {
        'exoplanets': cleaned_data[:100],  # Top 100
        'total': len(cleaned_data[:100]),
        'total_available': len(cleaned_data)
    }
    
    with open(api_dir / "exoplanets.json", 'w') as f:
        json.dump(api_response, f, indent=2)
    
    # Create different size variants
    for limit in [50, 200, 500]:
        variant_response = {
            'exoplanets': cleaned_data[:limit],
            'total': len(cleaned_data[:limit]),
            'total_available': len(cleaned_data)
        }
        with open(api_dir / f"exoplanets_{limit}.json", 'w') as f:
            json.dump(variant_response, f, indent=2)
    
    con.close()
    print(f"✅ Static API files created")

def main():
    """Export data to all formats"""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        print("   Please run dataPull.py first to create the database.")
        return
    
    # Create export directory
    EXPORT_DIR.mkdir(exist_ok=True)
    
    print("🚀 Starting data export...")
    print(f"   Source: {DB_PATH}")
    print(f"   Target: {EXPORT_DIR}")
    print()
    
    # Export to all formats
    export_to_json()
    export_to_csv()
    export_to_parquet()
    export_to_sqlite()
    create_static_api_files()
    
    print()
    print("🎉 Export complete! Files created:")
    print(f"   📁 {EXPORT_DIR}")
    print("   ├── 📄 json/          - JSON files for web apps")
    print("   ├── 📊 csv/           - CSV files for data analysis")
    print("   ├── 🗜️ parquet/       - Parquet files for big data")
    print("   ├── 🗃️ sqlite/        - SQLite database for portability")
    print("   └── 🌐 api/           - Static API-like JSON responses")
    print()
    print("💡 Usage suggestions:")
    print("   • Use JSON files for static websites")
    print("   • Use SQLite for portable web apps")
    print("   • Use CSV for data sharing")
    print("   • Use API files to replace live database calls")

if __name__ == "__main__":
    main() 