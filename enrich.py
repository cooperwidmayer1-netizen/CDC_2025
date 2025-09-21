#!/usr/bin/env python3
"""
enrich.py – Enhanced Exoplanet Description Generator
- Detailed 3-4 sentence descriptions using GPT-4o-mini
- Comprehensive planet information context
- Higher quality scientific explanations
- Balanced speed and detail optimization
- 25 concurrent requests for stability

Note: This script was primarily created using AI assistance.
"""

import os, re, asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
import time
import random

# Environment & clients
load_dotenv()
OAI_KEY = os.getenv("OPENAI_API_KEY")

# Configuration
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "exoplanets.duckdb"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable OpenAI HTTP request logging for speed
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

oai = AsyncOpenAI(api_key=OAI_KEY)

# Enhanced prompt for longer, more detailed descriptions
SYSTEM_PROMPT = """You are a space exploration expert writing engaging and detailed exoplanet descriptions. Create a vivid 3-4 sentence description that:

1. Opens with an evocative description of the planet's environment and key characteristics
2. Explains what makes this world unique using specific scientific details
3. Discusses its habitability prospects or interesting features

Use accessible language that conveys scientific significance while being exciting and discovery-focused. Include specific measurements and comparisons to Earth when relevant."""

def calculate_habitability_score(planet):
    """Fast habitability calculation with robust error handling"""
    try:
        score = 0
        
        # Insolation (0-40)
        if planet.get('pl_insol') is not None and not pd.isna(planet['pl_insol']):
            insolation = float(planet['pl_insol'])
            if 0.3 <= insolation <= 1.5:
                if insolation <= 1.0:
                    score += 40 * insolation
                else:
                    score += 40 * (1.5 - insolation) / 0.5
        
        # Radius (0-30)
        if planet.get('pl_rade') is not None and not pd.isna(planet['pl_rade']):
            radius = float(planet['pl_rade'])
            if 0.8 <= radius <= 1.4:
                if radius <= 1.0:
                    score += 30 * radius
                else:
                    score += 30 * (1.4 - radius) / 0.4
        
        # Temperature (0-20)
        if planet.get('pl_eqt') is not None and not pd.isna(planet['pl_eqt']):
            temp = float(planet['pl_eqt'])
            if 250 <= temp <= 350:
                if temp <= 300:
                    score += 20 * (temp - 250) / 50
                else:
                    score += 20 * (350 - temp) / 50
        
        # Mass (0-10)
        if planet.get('pl_bmasse') is not None and not pd.isna(planet['pl_bmasse']):
            mass = float(planet['pl_bmasse'])
            if 0.5 <= mass <= 2.0:
                if mass <= 1.0:
                    score += 10 * mass
                else:
                    score += 10 * (2.0 - mass)
        
        return round(max(0, score), 1)
    except (ValueError, TypeError):
        return 0.0

async def generate_description_fast(planet_data: Dict[str, Any], sem: asyncio.Semaphore, retries: int = 3) -> Optional[Dict[str, Any]]:
    """ULTRA-FAST description generation with exponential backoff"""
    async with sem:
        for attempt in range(retries):
            try:
                # Build minimal user message
                planet_name = planet_data.get('pl_name', 'Unknown')
                score = planet_data.get('habitability_score', 0)
                category = planet_data.get('habitability_category', 'Unknown')
                
                # Include distance if available
                distance_info = ""
                if planet_data.get('sy_dist'):
                    ly = planet_data['sy_dist'] * 3.26156
                    distance_info = f" | Distance: {ly:.1f} light-years"
                
                # ULTRA-SHORT user message for speed
                user_msg = f"{planet_name}: {score}/100 habitability ({category}), {planet_data.get('pl_rade', '?')}R⊕, {planet_data.get('pl_eqt', '?')}K{distance_info}"
                
                # Use GPT-3.5-turbo for speed and cost
                response = await oai.chat.completions.create(
                    model="gpt-3.5-turbo",  # Much faster than GPT-4
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=100,  # Shorter responses
                    temperature=0.7
                )
                
                description = response.choices[0].message.content.strip()
                
                return {
                    'pl_name': planet_name,
                    'description': description,
                    'habitability_score': score,
                    'habitability_category': category,
                    'habitability_color': planet_data.get('habitability_color', '#666666'),
                    'habitability_icon': planet_data.get('habitability_icon', 'fas fa-question')
                }
                
            except OpenAIError as e:
                if attempt < retries - 1:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed for {planet_name}, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to generate description for {planet_name} after {retries} attempts: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error for {planet_name}: {e}")
                return None

async def generate_description_detailed(planet_data: Dict[str, Any], sem: asyncio.Semaphore, retries: int = 3) -> Optional[Dict[str, Any]]:
    """Enhanced description generation with detailed information"""
    async with sem:
        for attempt in range(retries):
            try:
                # Build comprehensive user message
                planet_name = planet_data.get('pl_name', 'Unknown')
                score = planet_data.get('habitability_score', 0)
                category = planet_data.get('habitability_category', 'Unknown')
                
                # Gather detailed planet information
                radius = planet_data.get('pl_rade', 'Unknown')
                temp = planet_data.get('pl_eqt', 'Unknown')
                insolation = planet_data.get('pl_insol', 'Unknown')
                mass_earth = planet_data.get('pl_bmasse', 'Unknown')
                mass_jupiter = planet_data.get('pl_bmassj', 'Unknown')
                orbital_period = planet_data.get('pl_orbper', 'Unknown')
                host_star = planet_data.get('hostname', 'Unknown')
                discovery_year = planet_data.get('disc_year', 'Unknown')
                
                # Distance information
                distance_info = ""
                if planet_data.get('sy_dist'):
                    ly = planet_data['sy_dist'] * 3.26156
                    pc = planet_data['sy_dist']
                    distance_info = f" | Distance: {ly:.1f} light-years ({pc:.1f} parsecs)"
                
                # Stellar information
                stellar_info = ""
                if planet_data.get('st_teff'):
                    stellar_temp = planet_data['st_teff']
                    stellar_info = f" | Host star temp: {stellar_temp:.0f}K"
                
                # Orbital information
                orbital_info = ""
                if planet_data.get('pl_orbsmax'):
                    orbit_au = planet_data['pl_orbsmax']
                    orbital_info = f" | Orbit: {orbit_au:.3f} AU"
                
                # Comprehensive user message with rich context
                user_msg = f"""Planet: {planet_name} 
Host Star: {host_star}
Habitability: {score}/100 ({category})
Physical Properties: Radius {radius}R⊕, Mass {mass_earth}M⊕, Temperature {temp}K
Orbital Properties: Period {orbital_period} days{orbital_info}
Stellar Flux: {insolation} times Earth's
Discovery Year: {discovery_year}{distance_info}{stellar_info}"""
                
                # Use GPT-4 for higher quality longer descriptions
                response = await oai.chat.completions.create(
                    model="gpt-4o-mini",  # Better quality for detailed descriptions
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=400,  # Much longer responses
                    temperature=0.8  # Slightly more creative
                )
                
                description = response.choices[0].message.content.strip()
                
                return {
                    'pl_name': planet_name,
                    'description': description,
                    'habitability_score': score,
                    'habitability_category': category,
                    'habitability_color': planet_data.get('habitability_color', '#666666'),
                    'habitability_icon': planet_data.get('habitability_icon', 'fas fa-question')
                }
                
            except OpenAIError as e:
                if attempt < retries - 1:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed for {planet_name}, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to generate description for {planet_name} after {retries} attempts: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error for {planet_name}: {e}")
                return None

async def ultra_fast_batch_update(enriched_planets: List[Dict[str, Any]]) -> None:
    """REMOVED - using inline updates for maximum speed"""
    pass

async def enrich_all_planets() -> None:
    """Enhanced enrichment process with detailed descriptions"""
    start_time = time.time()
    
    # Get all planets and ensure columns exist
    con = duckdb.connect(str(DB_PATH))
    try:
        # Ensure columns exist
        con.execute("ALTER TABLE exo.raw_ps ADD COLUMN IF NOT EXISTS description TEXT")
        con.execute("ALTER TABLE exo.raw_ps ADD COLUMN IF NOT EXISTS habitability_score INTEGER")
        con.execute("ALTER TABLE exo.raw_ps ADD COLUMN IF NOT EXISTS habitability_category TEXT")
        con.execute("ALTER TABLE exo.raw_ps ADD COLUMN IF NOT EXISTS habitability_color TEXT")
        con.execute("ALTER TABLE exo.raw_ps ADD COLUMN IF NOT EXISTS habitability_icon TEXT")
        
        planets = con.execute("SELECT * FROM exo.raw_ps").fetchdf().to_dict('records')
        logger.info(f"🚀 ENHANCED MODE: Processing {len(planets)} planets with detailed descriptions")
        
        # Add habitability calculations
        for planet in planets:
            score = calculate_habitability_score(planet)
            category_index = min(4, max(0, int(score // 20))) if score is not None and not pd.isna(score) else 0
            categories = ["Not Habitable", "Unlikely Habitable", "Marginally Habitable", "Potentially Habitable", "Highly Habitable"]
            colors = ["#ff3366", "#ff6600", "#ffaa00", "#00ff88", "#00ff88"]
            icons = ["fas fa-times-circle", "fas fa-exclamation-triangle", "fas fa-leaf", "fas fa-seedling", "fas fa-seedling"]
            
            planet['habitability_score'] = score
            planet['habitability_category'] = categories[category_index]
            planet['habitability_color'] = colors[category_index]
            planet['habitability_icon'] = icons[category_index]
            
    finally:
        con.close()
    
    # Reduced concurrency for longer, higher-quality descriptions
    sem = asyncio.Semaphore(25)  # Lower concurrency for GPT-4 and longer responses
    
    # Smaller batches for better quality control
    batch_size = 50
    
    for i in range(0, len(planets), batch_size):
        batch = planets[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(planets) + batch_size - 1) // batch_size
        
        logger.info(f"📝 Enhanced batch {batch_num}/{total_batches} ({len(batch)} planets)")
        
        # Process with detailed generation
        tasks = [generate_description_detailed(planet, sem) for planet in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        if successful:
            logger.info(f"✅ Generated {len(successful)} detailed descriptions in batch {batch_num}")
            
            # Update database with descriptions
            con = duckdb.connect(str(DB_PATH))
            try:
                for planet in successful:
                    con.execute("""
                        UPDATE exo.raw_ps 
                        SET description = ?, 
                            habitability_score = ?, 
                            habitability_category = ?, 
                            habitability_color = ?, 
                            habitability_icon = ?
                        WHERE pl_name = ?
                    """, (
                        planet['description'],
                        planet['habitability_score'],
                        planet['habitability_category'], 
                        planet['habitability_color'],
                        planet['habitability_icon'],
                        planet['pl_name']
                    ))
            finally:
                con.close()
        else:
            logger.warning(f"❌ No successful results in batch {batch_num}")
        
        # Longer pause between batches for GPT-4 rate limits
        if i + batch_size < len(planets):
            await asyncio.sleep(0.5)  # Longer delay for detailed processing
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 ENHANCED enrichment complete! Processed {len(planets)} planets with detailed descriptions in {elapsed:.1f}s ({len(planets)/elapsed:.1f} planets/sec)")

def main():
    """Main execution with enhanced detailed processing"""
    if not OAI_KEY:
        logger.error("❌ OPENAI_API_KEY not found in environment")
        return
        
    logger.info("🚀 ENHANCED MODE: Generating detailed exoplanet descriptions")
    asyncio.run(enrich_all_planets())

if __name__ == "__main__":
    main() 