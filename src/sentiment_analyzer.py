"""
Sentiment Analyzer
Computes sentiment scores from lyrics using VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""

import logging
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Optional
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of lyrics using NLTK's VADER."""
    
    def __init__(self):
        """Initialize sentiment analyzer and download required NLTK data."""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze (lyrics)
            
        Returns:
            Dictionary with sentiment scores:
            - neg: Negative score (0.0 to 1.0)
            - neu: Neutral score (0.0 to 1.0)
            - pos: Positive score (0.0 to 1.0)
            - compound: Compound score (-1.0 to 1.0)
        """
        if not text or not text.strip():
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        scores = self.sia.polarity_scores(text)
        return scores
    
    def analyze_lyrics_file(self, filepath: str) -> Dict[str, float]:
        """
        Analyze sentiment of lyrics from a file.
        
        Args:
            filepath: Path to lyrics text file
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.analyze_text(text)
        except Exception as e:
            logger.error("Error analyzing file %s: %s", filepath, e)
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get simplified sentiment score.
        
        Args:
            text: Text to analyze
            
        Returns:
            Compound sentiment score (-1.0 to 1.0)
            Negative: -1.0 to -0.05
            Neutral: -0.05 to 0.05
            Positive: 0.05 to 1.0
        """
        scores = self.analyze_text(text)
        return scores['compound']
    
    def analyze_batch(self, lyrics_dict: Dict[str, Optional[str]], n_jobs: int = -1) -> pd.DataFrame:
        """
        Analyze sentiment for multiple tracks (parallelized).
        
        Args:
            lyrics_dict: Dictionary mapping track names to lyrics text
            n_jobs: Number of parallel jobs (-1 = all CPU cores)
            
        Returns:
            DataFrame with sentiment scores for each track
        """
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_jobs = max(1, min(n_jobs, len(lyrics_dict)))
        
        items = list(lyrics_dict.items())
        
        with Pool(processes=n_jobs) as pool:
            results_list = list(tqdm(
                pool.imap(_analyze_sentiment_worker, items),
                total=len(items),
                desc="  Analyzing sentiment"
            ))
        
        return pd.DataFrame(results_list)


def _analyze_sentiment_worker(item):
    """
    Worker function for sentiment analysis (must be at module level for pickling).
    
    Args:
        item: Tuple of (track_name, lyrics_text)
        
    Returns:
        Dictionary with sentiment scores
    """
    track_name, lyrics = item
    
    try:
        analyzer = SentimentAnalyzer()
        if lyrics:
            scores = analyzer.analyze_text(lyrics)
        else:
            scores = {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    except Exception:
        scores = {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    return {
        'track_name': track_name,
        'sentiment_compound': scores['compound'],
        'sentiment_pos': scores['pos'],
        'sentiment_neu': scores['neu'],
        'sentiment_neg': scores['neg']
    }
