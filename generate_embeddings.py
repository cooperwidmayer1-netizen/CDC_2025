#!/usr/bin/env python3
"""
generate_embeddings.py - Generate embeddings for planet descriptions
This script creates embeddings for all planet descriptions to enable semantic search

Note: This script was primarily created using AI assistance.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
import time

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

# Disable OpenAI HTTP request logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

oai = AsyncOpenAI(api_key=OAI_KEY)

async def generate_embedding(text: str, semaphore: asyncio.Semaphore, retries: int = 3) -> List[float]:
    """Generate embedding for text with retry logic"""
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await oai.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            except OpenAIError as e:
                if attempt < retries - 1:
                    delay = (2 ** attempt) + 0.1
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to generate embedding after {retries} attempts: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                return None

async def generate_all_embeddings():
    """Generate embeddings for all planet descriptions"""
    start_time = time.time()
    
    # Connect to database
    con = duckdb.connect(str(DB_PATH))
    
    try:
        # Get all planets with descriptions but no embeddings
        planets = con.execute("""
            SELECT pl_name, description 
            FROM exo.raw_ps 
            WHERE description IS NOT NULL 
            AND (embedding IS NULL OR array_length(embedding) = 0)
        """).fetchall()
        
        logger.info(f"🚀 Found {len(planets)} planets needing embeddings")
        
        if len(planets) == 0:
            logger.info("✅ All planets already have embeddings!")
            return
        
        # Process ALL planets concurrently with high concurrency
        semaphore = asyncio.Semaphore(50)  # Much higher concurrency
        
        logger.info(f"🚀 Generating embeddings for all {len(planets)} planets concurrently")
        
        # Create tasks for ALL planets at once
        all_tasks = []
        for planet_name, description in planets:
            if description and len(description.strip()) > 0:
                task = generate_embedding(description, semaphore)
                all_tasks.append((planet_name, task))
        
        # Process in chunks for database updates
        chunk_size = 100
        total_processed = 0
        
        for i in range(0, len(all_tasks), chunk_size):
            chunk = all_tasks[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(all_tasks) + chunk_size - 1) // chunk_size
            
            logger.info(f"⚡ Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} planets)")
            
            # Wait for this chunk's embeddings
            results = []
            for planet_name, task in chunk:
                embedding = await task
                if embedding is not None:
                    results.append((planet_name, embedding))
                total_processed += 1
                
                # Progress update every 50 planets
                if total_processed % 50 == 0:
                    logger.info(f"📊 Progress: {total_processed}/{len(planets)} planets processed")
            
            # Update database with this chunk's results
            if results:
                logger.info(f"💾 Saving {len(results)} embeddings to database")
                for planet_name, embedding in results:
                    con.execute("""
                        UPDATE exo.raw_ps 
                        SET embedding = ? 
                        WHERE pl_name = ?
                    """, (embedding, planet_name))
                
                logger.info(f"✅ Chunk {chunk_num} saved ({len(results)} embeddings)")
            else:
                logger.warning(f"❌ No successful embeddings in chunk {chunk_num}")
    
    finally:
        con.close()
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 Embedding generation complete! Processed {len(planets)} planets in {elapsed:.1f}s")

def main():
    """Main execution"""
    if not OAI_KEY:
        logger.error("❌ OPENAI_API_KEY not found in environment")
        return
    
    if not DB_PATH.exists():
        logger.error(f"❌ Database not found at {DB_PATH}")
        return
    
    logger.info("🚀 Starting embedding generation for semantic search")
    asyncio.run(generate_all_embeddings())

if __name__ == "__main__":
    main() 