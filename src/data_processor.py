"""
Data Processor
Combines Spotify audio features with lyrical sentiment analysis
and prepares the final dataset used for visualization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MoodDataProcessor:
    """Processes and combines mood-related data for visualization."""
    
    @staticmethod
    def combine_features(
        spotify_df: pd.DataFrame,
        lyrics_dict: Dict[str, Optional[str]],
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine Spotify features with VADER sentiment analysis.
        
        Args:
            spotify_df: DataFrame with Spotify tracks and audio features
            lyrics_dict: Dictionary mapping track names to lyrics
            sentiment_df: DataFrame with lyrics sentiment scores
            
        Returns:
            Combined DataFrame with valence, arousal, and lyrics sentiment
        """
        spotify_df['has_lyrics'] = spotify_df['name'].apply(
            lambda x: lyrics_dict.get(x) is not None if x in lyrics_dict else False
        )
        
        df = pd.merge(
            spotify_df,
            sentiment_df,
            left_on='name',
            right_on='track_name',
            how='left'
        )
        # if missing, fill with 0.0
        sentiment_cols = ['sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg']
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        return df
    
    @staticmethod
    def create_mood_array(df: pd.DataFrame) -> np.ndarray:
        """
        Create a mood array suitable for visualization.
        
        Args:
            df: DataFrame with valence, arousal, and sentiment columns
            
        Returns:
            NumPy array with shape (n_tracks, 4) containing:
            [position, valence, arousal, sentiment]
        """
        mood_data = df[['position', 'valence', 'arousal', 'sentiment_compound']].values
        return mood_data
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalize specified columns to 0-1 range (min-max normalization).
        
        Args:
            df: DataFrame
            columns: List of column names to normalize
            
        Returns:
            DataFrame with normalized columns
        """
        df_normalized = df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        return df_normalized
    
    @staticmethod
    def normalize_valence_arousal(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize valence and arousal values to 0-1 range (min-max normalization).
        
        Similar to how sentiment is normalized. This scales values so min=0 and max=1.
        Formula: (x - min) / (max - min)
        
        Args:
            df: DataFrame with 'valence' and 'arousal' columns
            
        Returns:
            DataFrame with normalized valence and arousal columns (0-1 range)
        """
        df_normalized = df.copy()
        
        # Normalize valence to 0-1 range
        if 'valence' in df_normalized.columns:
            valence_min = df_normalized['valence'].min()
            valence_max = df_normalized['valence'].max()
            valence_range = valence_max - valence_min
            if valence_range > 0:
                df_normalized['valence'] = (df_normalized['valence'] - valence_min) / valence_range
            else:
                # If all values are the same, set to 0.5 (middle)
                df_normalized['valence'] = 0.5
        
        # Normalize arousal to 0-1 range
        if 'arousal' in df_normalized.columns:
            arousal_min = df_normalized['arousal'].min()
            arousal_max = df_normalized['arousal'].max()
            arousal_range = arousal_max - arousal_min
            if arousal_range > 0:
                df_normalized['arousal'] = (df_normalized['arousal'] - arousal_min) / arousal_range
            else:
                # If all values are the same, set to 0.5 (middle)
                df_normalized['arousal'] = 0.5
        
        return df_normalized
    
    @staticmethod
    def save_mood_data(df: pd.DataFrame, output_path: str):
        """
        Save mood data to CSV.
        
        Args:
            df: DataFrame to save
            output_path: Path to output CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Mood data saved to %s", output_path)
    
    @staticmethod
    def load_mood_data(input_path: str) -> pd.DataFrame:
        """
        Load mood data from CSV.
        
        Args:
            input_path: Path to input CSV file
            
        Returns:
            DataFrame with mood data
        """
        return pd.read_csv(input_path)
    
    @staticmethod
    def get_color_mapping(valence: float, arousal: float, sentiment: float) -> tuple:
        """
        Map valence, arousal, and sentiment to RGB color.
        
        Mapping strategy:
        - Valence -> Hue (low valence = cool/blue, high valence = warm/red)
        - Arousal -> Saturation (low arousal = desaturated, high arousal = saturated)
        - Sentiment -> Brightness/Value (negative = darker, positive = brighter)
        
        Args:
            valence: 0.0 to 1.0
            arousal: 0.0 to 1.0
            sentiment: -1.0 to 1.0
            
        Returns:
            RGB tuple (0-255, 0-255, 0-255)
        """
        import colorsys
        
        # Map valence to hue (0-1 range, where 0=red, 0.5=cyan, 1.0=red)
        # use 0.0-0.7 range to avoid wrapping
        # Low valence (sad) = cool colors (blue/cyan), High valence (happy) = warm colors (red/orange)
        hue = 0.7 - (valence * 0.7)
        
        saturation = arousal

        value = 0.3 + ((sentiment + 1) / 2) * 0.7
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        return tuple(int(c * 255) for c in rgb)
    
    @staticmethod
    def create_color_timeline(df: pd.DataFrame) -> list:
        """
        Create a color timeline for the entire playlist.
        
        Args:
            df: DataFrame with valence, arousal, and sentiment columns
            
        Returns:
            List of RGB tuples, one per track
        """
        colors = []
        
        for _, row in df.iterrows():
            valence = row.get('valence', 0.5)
            arousal = row.get('arousal', 0.5)
            sentiment = row.get('sentiment_compound', 0.0)
            
            color = MoodDataProcessor.get_color_mapping(valence, arousal, sentiment)
            colors.append(color)
        
        return colors
    
    @staticmethod
    def generate_summary_stats(df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the playlist.
        
        Args:
            df: DataFrame with mood data
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_tracks': len(df),
            'avg_valence': df['valence'].mean(),
            'avg_arousal': df['arousal'].mean(),
            'avg_sentiment': df['sentiment_compound'].mean(),
            'tracks_with_lyrics': df['has_lyrics'].sum() if 'has_lyrics' in df.columns else 0,
            'valence_std': df['valence'].std(),
            'arousal_std': df['arousal'].std(),
            'sentiment_std': df['sentiment_compound'].std(),
        }

        # project to something that resembles the Russell Circumplex Model        
        energetic_positive = ((df['valence'] > 0.5) & (df['arousal'] > 0.5)).sum()
        calm_positive = ((df['valence'] > 0.5) & (df['arousal'] <= 0.5)).sum()
        energetic_negative = ((df['valence'] <= 0.5) & (df['arousal'] > 0.5)).sum()
        calm_negative = ((df['valence'] <= 0.5) & (df['arousal'] <= 0.5)).sum()
        
        # find the dominant quadrant
        quadrant_counts = {
            'Energetic & Positive': energetic_positive,
            'Calm & Positive': calm_positive,
            'Energetic & Negative': energetic_negative,
            'Calm & Negative': calm_negative
        }
        stats['overall_mood'] = max(quadrant_counts, key=quadrant_counts.get)
        
        return stats
