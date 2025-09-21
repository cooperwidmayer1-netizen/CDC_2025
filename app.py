#!/usr/bin/env python3
"""
Space-themed Flask app for exoplanet data visualization with habitability scoring and semantic search
"""

from flask import Flask, render_template, jsonify, request
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configuration
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "exoplanets.duckdb"
OAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
oai = OpenAI(api_key=OAI_KEY) if OAI_KEY else None

def get_db_connection():
    """Get DuckDB connection"""
    if DB_PATH.exists():
        return duckdb.connect(str(DB_PATH))
    return None

def clean_data_for_json(data):
    """Clean data to make it JSON serializable - fixed for arrays"""
    if isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        # Handle numpy arrays (like embeddings)
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        if pd.isna(data) or np.isnan(data):
            return None
        return float(data) if isinstance(data, np.floating) else int(data)
    elif pd.isna(data):
        return None
    else:
        return data

def calculate_habitability_score(planet):
    """Calculate habitability score based on key parameters"""
    score = 0
    
    # Insolation score (0-40 points)
    if planet.get('pl_insol') is not None and not pd.isna(planet['pl_insol']):
        insolation = planet['pl_insol']
        if 0.3 <= insolation <= 1.04:
            if insolation <= 1.0:
                insolation_score = 40 * (insolation / 1.0)
            else:
                insolation_score = 40 * (1.5 - insolation) / 0.5
            score += max(0, insolation_score)
    
    # Radius score (0-30 points)
    if planet.get('pl_rade') is not None and not pd.isna(planet['pl_rade']):
        radius = planet['pl_rade']
        if 0.8 <= radius <= 1.4:
            if radius <= 1.0:
                radius_score = 30 * (radius / 1.0)
            else:
                radius_score = 30 * (1.4 - radius) / 0.4
            score += max(0, radius_score)
    
    # Temperature score (0-20 points)
    if planet.get('pl_eqt') is not None and not pd.isna(planet['pl_eqt']):
        temp = planet['pl_eqt']
        if 250 <= temp <= 350:
            if temp <= 300:
                temp_score = 20 * (temp - 250) / 50
            else:
                temp_score = 20 * (350 - temp) / 50
            score += max(0, temp_score)
    
    # Mass bonus (0-10 points)
    if planet.get('pl_bmasse') is not None and not pd.isna(planet['pl_bmasse']):
        mass = planet['pl_bmasse']
        if 0.5 <= mass <= 2.0:
            if mass <= 1.0:
                mass_score = 10 * (mass / 1.0)
            else:
                mass_score = 10 * (2.0 - mass) / 1.0
            score += max(0, mass_score)
    
    return round(score, 1)

def get_habitability_category(score):
    """Get habitability category based on score"""
    if score >= 80:
        return {"category": "Highly Habitable", "color": "#00ff88", "icon": "fas fa-seedling"}
    elif score >= 60:
        return {"category": "Potentially Habitable", "color": "#ffaa00", "icon": "fas fa-leaf"}
    elif score >= 40:
        return {"category": "Marginally Habitable", "color": "#ff6600", "icon": "fas fa-exclamation-triangle"}
    elif score >= 20:
        return {"category": "Unlikely Habitable", "color": "#ff3366", "icon": "fas fa-times-circle"}
    else:
        return {"category": "Not Habitable", "color": "#666666", "icon": "fas fa-ban"}

def get_query_embedding(query_text):
    """Get embedding for search query"""
    if not oai:
        return None
    
    try:
        response = oai.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0
    
    a = np.array(a)
    b = np.array(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

@app.route('/')
def index():
    """Main page - Exoplanets search"""
    return render_template('exoplanets.html')

@app.route('/api/exoplanets')
def get_exoplanets():
    """API endpoint to get exoplanet data with habitability scores and optional semantic search"""
    con = get_db_connection()
    if not con:
        return jsonify({'error': 'Database not found. Please run dataPull.py first.'}), 404
    
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        search = request.args.get('search', '', type=str)
        semantic_search = request.args.get('semantic', 'false').lower() == 'true'
        
        # Build query - get ALL data first, then we'll sort and limit in Python
        query = "SELECT * FROM exo.raw_ps"
        params = []
        
        if search and not semantic_search:
            # Traditional text search
            query += " WHERE pl_name ILIKE ? OR hostname ILIKE ?"
            params.extend([f'%{search}%', f'%{search}%'])
        
        # Execute query to get ALL matching records
        df = con.execute(query, params).df()
        con.close()
        
        # Convert to JSON-serializable format and add habitability scores
        data = df.to_dict('records')
        
        # Add habitability scores to each planet
        for planet in data:
            score = calculate_habitability_score(planet)
            habitability_info = get_habitability_category(score)
            
            planet['habitability_score'] = score
            planet['habitability_category'] = habitability_info['category']
            planet['habitability_color'] = habitability_info['color']
            planet['habitability_icon'] = habitability_info['icon']
        
        # Handle semantic search
        if search and semantic_search and oai:
            query_embedding = get_query_embedding(search)
            if query_embedding is not None and len(query_embedding) > 0:
                # Calculate similarity scores for planets with embeddings
                for planet in data:
                    embedding = planet.get('embedding')
                    if embedding is not None and len(embedding) > 0:
                        similarity = cosine_similarity(query_embedding, embedding)
                        planet['similarity_score'] = similarity
                    else:
                        planet['similarity_score'] = 0
                
                # Sort by similarity score (highest first)
                data.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            else:
                # Fallback to habitability sorting if embedding fails
                data.sort(key=lambda x: x['habitability_score'] or 0, reverse=True)
        else:
            # Sort by habitability score (highest first)
            data.sort(key=lambda x: x['habitability_score'] or 0, reverse=True)
        
        # Apply limit AFTER sorting
        limited_data = data[:limit]
        
        cleaned_data = clean_data_for_json(limited_data)
        
        return jsonify({
            'exoplanets': cleaned_data,
            'total': len(limited_data),
            'total_available': len(data),
            'search_type': 'semantic' if (search and semantic_search) else 'traditional'
        })
    
    except Exception as e:
        if con:
            con.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/perfect-planet')
def get_perfect_planet():
    """API endpoint to find planets matching user-specified criteria"""
    con = get_db_connection()
    if not con:
        return jsonify({'error': 'Database not found. Please run dataPull.py first.'}), 404
    
    try:
        # Get filter parameters from query string
        min_radius = request.args.get('min_radius', type=float)
        max_radius = request.args.get('max_radius', type=float)
        min_mass = request.args.get('min_mass', type=float)
        max_mass = request.args.get('max_mass', type=float)
        min_period = request.args.get('min_period', type=float)
        max_period = request.args.get('max_period', type=float)
        min_insolation = request.args.get('min_insolation', type=float)
        max_insolation = request.args.get('max_insolation', type=float)
        min_density = request.args.get('min_density', type=float)
        max_density = request.args.get('max_density', type=float)
        min_orbit = request.args.get('min_orbit', type=float)
        max_orbit = request.args.get('max_orbit', type=float)
        min_eccentricity = request.args.get('min_eccentricity', type=float)
        max_eccentricity = request.args.get('max_eccentricity', type=float)
        min_distance = request.args.get('min_distance', type=float)
        max_distance = request.args.get('max_distance', type=float)
        min_habitability = request.args.get('min_habitability', type=int)
        limit = request.args.get('limit', 100, type=int)
        
        # Build dynamic WHERE clause
        where_conditions = []
        params = []
        
        if min_radius is not None:
            where_conditions.append("pl_rade >= ?")
            params.append(min_radius)
        if max_radius is not None:
            where_conditions.append("pl_rade <= ?")
            params.append(max_radius)
            
        if min_mass is not None:
            where_conditions.append("pl_bmasse >= ?")
            params.append(min_mass)
        if max_mass is not None:
            where_conditions.append("pl_bmasse <= ?")
            params.append(max_mass)
            
        if min_period is not None:
            where_conditions.append("pl_orbper >= ?")
            params.append(min_period)
        if max_period is not None:
            where_conditions.append("pl_orbper <= ?")
            params.append(max_period)
            
        if min_density is not None:
            where_conditions.append("pl_dens >= ?")
            params.append(min_density)
        if max_density is not None:
            where_conditions.append("pl_dens <= ?")
            params.append(max_density)
            
        if min_orbit is not None:
            where_conditions.append("pl_orbsmax >= ?")
            params.append(min_orbit)
        if max_orbit is not None:
            where_conditions.append("pl_orbsmax <= ?")
            params.append(max_orbit)
            
        if min_eccentricity is not None:
            where_conditions.append("pl_orbeccen >= ?")
            params.append(min_eccentricity)
        if max_eccentricity is not None:
            where_conditions.append("pl_orbeccen <= ?")
            params.append(max_eccentricity)
            
        if min_insolation is not None:
            where_conditions.append("pl_insol >= ?")
            params.append(min_insolation)
        if max_insolation is not None:
            where_conditions.append("pl_insol <= ?")
            params.append(max_insolation)
            
        if min_distance is not None:
            where_conditions.append("sy_dist >= ?")
            params.append(min_distance)
        if max_distance is not None:
            where_conditions.append("sy_dist <= ?")
            params.append(max_distance)
        
        # Build query
        query = "SELECT * FROM exo.raw_ps"
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        # Execute query
        df = con.execute(query, params).df()
        con.close()
        
        # Convert to JSON-serializable format and add habitability scores
        data = df.to_dict('records')
        
        # Add habitability scores and filter by habitability if requested
        filtered_data = []
        for planet in data:
            score = calculate_habitability_score(planet)
            habitability_info = get_habitability_category(score)
            
            planet['habitability_score'] = score
            planet['habitability_category'] = habitability_info['category']
            planet['habitability_color'] = habitability_info['color']
            planet['habitability_icon'] = habitability_info['icon']
            
            # Apply habitability filter if specified
            if min_habitability is None or score >= min_habitability:
                filtered_data.append(planet)
        
        # Sort by habitability score (highest first)
        filtered_data.sort(key=lambda x: x['habitability_score'] or 0, reverse=True)
        
        # Apply limit
        limited_data = filtered_data[:limit]
        
        cleaned_data = clean_data_for_json(limited_data)
        
        return jsonify({
            'exoplanets': cleaned_data,
            'total': len(limited_data),
            'total_available': len(filtered_data),
            'filters_applied': len(where_conditions) > 0 or min_habitability is not None
        })
    
    except Exception as e:
        if con:
            con.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-ranges')
def get_data_ranges():
    """API endpoint to get data ranges for perfect planet builder"""
    con = get_db_connection()
    if not con:
        return jsonify({'error': 'Database not found'}), 404
    
    try:
        ranges_query = """
        SELECT 
            'radius' as field,
            MIN(pl_rade) as min_val,
            MAX(pl_rade) as max_val,
            AVG(pl_rade) as avg_val,
            COUNT(pl_rade) as count_non_null
        FROM exo.raw_ps
        WHERE pl_rade IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'mass' as field,
            MIN(pl_bmasse) as min_val,
            MAX(pl_bmasse) as max_val,
            AVG(pl_bmasse) as avg_val,
            COUNT(pl_bmasse) as count_non_null
        FROM exo.raw_ps
        WHERE pl_bmasse IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'density' as field,
            MIN(pl_dens) as min_val,
            MAX(pl_dens) as max_val,
            AVG(pl_dens) as avg_val,
            COUNT(pl_dens) as count_non_null
        FROM exo.raw_ps
        WHERE pl_dens IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'orbit' as field,
            MIN(pl_orbsmax) as min_val,
            MAX(pl_orbsmax) as max_val,
            AVG(pl_orbsmax) as avg_val,
            COUNT(pl_orbsmax) as count_non_null
        FROM exo.raw_ps
        WHERE pl_orbsmax IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'eccentricity' as field,
            MIN(pl_orbeccen) as min_val,
            MAX(pl_orbeccen) as max_val,
            AVG(pl_orbeccen) as avg_val,
            COUNT(pl_orbeccen) as count_non_null
        FROM exo.raw_ps
        WHERE pl_orbeccen IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'period' as field,
            MIN(pl_orbper) as min_val,
            MAX(pl_orbper) as max_val,
            AVG(pl_orbper) as avg_val,
            COUNT(pl_orbper) as count_non_null
        FROM exo.raw_ps
        WHERE pl_orbper IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'insolation' as field,
            MIN(pl_insol) as min_val,
            MAX(pl_insol) as max_val,
            AVG(pl_insol) as avg_val,
            COUNT(pl_insol) as count_non_null
        FROM exo.raw_ps
        WHERE pl_insol IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'distance' as field,
            MIN(sy_dist) as min_val,
            MAX(sy_dist) as max_val,
            AVG(sy_dist) as avg_val,
            COUNT(sy_dist) as count_non_null
        FROM exo.raw_ps
        WHERE sy_dist IS NOT NULL
        """
        
        ranges_df = con.execute(ranges_query).df()
        ranges_data = ranges_df.to_dict('records')
        
        con.close()
        
        # Convert to a more usable format
        ranges = {}
        for row in ranges_data:
            field = row['field']
            ranges[field] = {
                'min': row['min_val'],
                'max': row['max_val'],
                'avg': row['avg_val'],
                'count': row['count_non_null']
            }
        
        cleaned_ranges = clean_data_for_json(ranges)
        
        return jsonify({
            'ranges': cleaned_ranges
        })
    
    except Exception as e:
        if con:
            con.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint to get exoplanet statistics"""
    con = get_db_connection()
    if not con:
        return jsonify({'error': 'Database not found'}), 404
    
    try:
        # Get basic statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_planets,
            COUNT(DISTINCT hostname) as total_stars,
            AVG(pl_rade) as avg_radius,
            AVG(pl_insol) as avg_insolation,
            MIN(disc_year) as first_discovery,
            MAX(disc_year) as latest_discovery
        FROM exo.raw_ps
        """
        
        stats_df = con.execute(stats_query).df()
        stats = stats_df.iloc[0].to_dict()
        
        # Get discovery timeline
        timeline_query = """
        SELECT disc_year, COUNT(*) as count
        FROM exo.raw_ps
        WHERE disc_year IS NOT NULL
        GROUP BY disc_year
        ORDER BY disc_year
        """
        
        timeline_df = con.execute(timeline_query).df()
        timeline = timeline_df.to_dict('records')
        
        con.close()
        
        # Clean the data for JSON serialization
        cleaned_stats = clean_data_for_json(stats)
        cleaned_timeline = clean_data_for_json(timeline)
        
        return jsonify({
            'stats': cleaned_stats,
            'timeline': cleaned_timeline
        })
    
    except Exception as e:
        if con:
            con.close()
        return jsonify({'error': str(e)}), 500

# Exoplanets page is now the main page at '/'

# Perfect Planet Builder is now integrated into the main page

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
