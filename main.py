"""
Main script for running the mood visualization pipeline.

Workflow:
1. Fetch playlist tracks from Spotify (metadata only)
2. Download audio from YouTube for each track
3. Analyze features of the audio to compute valence and arousal
4. Fetch lyrics for each track
5. Compute sentiment scores from lyrics
6. Combine all data into mood dataframe
7. Create visualizations and color mappings

Usage:
`python main.py "https://open.spotify.com/playlist/YOUR_PLAYLIST_ID"`

or 

`python main.py "YOUR_PLAYLIST_URL"`
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.spotify_client import SpotifyClient
from src.lyrics_fetcher import LyricsFetcher
from src.sentiment_analyzer import SentimentAnalyzer
from src.audio_analyzer import AudioAnalyzer
from src.data_processor import MoodDataProcessor
from src.visualizer import MoodVisualizer

logger = logging.getLogger(__name__)


def _track_key(artist_name: str, track_name: str) -> str:
    """Normalize track identity for global blocklist matching."""
    return f"{artist_name.strip().lower()}::{track_name.strip().lower()}"


def main(playlist_url: str, 
         skip_lyrics: bool = False, 
         skip_audio: bool = False,
         output_dir: str = "outputs",
         force_refresh: bool = False,
         n_jobs: int = -1):
    """
    Main workflow for Spotify mood visualization.
    
    Workflow:
    1. Fetch playlist tracks from Spotify (metadata only)
    2. Download audio from YouTube for each track
    3. Analyze audio to compute valence and arousal
    4. Fetch lyrics for each track
    5. Compute sentiment scores from lyrics
    6. Combine all data into mood dataframe
    7. Create visualizations and color mappings
    
    Args:
        playlist_url: Spotify playlist URL
        skip_lyrics: Skip lyrics fetching
        skip_audio: Skip audio download/analysis (will use default values)
        output_dir: Directory for outputs
        force_refresh: Force refresh playlist data (ignore cache)
        n_jobs: Number of parallel jobs/cores to use (-1 uses all cores)
    """
    load_dotenv()
    
    logger.info("SPOTIFY MOOD VISUALIZATION PIPELINE")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info("\n[1/6] Fetching playlist metadata from Spotify...")
    try:
        spotify_client = SpotifyClient()
        df_spotify = spotify_client.get_playlist_tracks_df(playlist_url, use_cache=not force_refresh)
        logger.info("  Found %s tracks", len(df_spotify))
    except Exception as e:
        logger.error("  Error: %s", e)
        return
    
    tracks = df_spotify[['name', 'artist', 'position']].to_dict('records')

    audio_analyzer = AudioAnalyzer()
    audio_features_dict = {}
    if not skip_audio:
        logger.info("\n[2/6] Downloading and analyzing audio from YouTube...")
        logger.info("  (This may take a while...")
        logger.info("  Note: Some downloads may fail due to YouTube restrictions.")
        logger.info("  Tip: Update yt-dlp if you see many 403 errors: pip install -U yt-dlp")
        try:
            audio_features_dict = audio_analyzer.batch_download_and_analyze(
                tracks, 
                output_dir="audio_cache",
                delay=2.0,  # Not used in parallel mode
                use_duration_matching=True,  # Filter results by duration to match youtube to spotify
                n_jobs=n_jobs
            )
            
            successful = sum(1 for v in audio_features_dict.values() if v is not None)
            failed = len(tracks) - successful
            logger.info("  Analyzed %s/%s tracks", successful, len(tracks))
            if failed > 0:
                logger.warning(
                    "  Warning: %s tracks failed - using default values (valence=0.5, arousal=0.5)",
                    failed,
                )
        except KeyboardInterrupt:
            logger.warning("\n  Warning: Interrupted by user")
            logger.info("  Continuing with partial results...")
        except Exception as e:
            logger.error("  Error: %s", e)
            logger.info("  Continuing with default values...")
    else:
        logger.info("\n[2/6] Skipping audio download/analysis...")
    
    # if there are any blocked tracks in the dict in audio analyzer, don't attempt to process
    blocked_keys = audio_analyzer.get_blocked_track_keys()
    if blocked_keys:
        blocked_mask = df_spotify.apply(
            lambda row: _track_key(row['artist'], row['name']) in blocked_keys,
            axis=1
        )
        blocked_count = int(blocked_mask.sum())
        if blocked_count > 0:
            df_spotify = df_spotify.loc[~blocked_mask].copy()
            tracks = df_spotify[['name', 'artist', 'position']].to_dict('records')
            logger.warning(
                "  Warning: Globally blocked tracks removed from downstream pipeline: %s",
                blocked_count,
            )
    
    def get_feature(track_name, feature, default):
        """Safely get feature from audio_features_dict, handling None values."""
        features = audio_features_dict.get(track_name)
        if features is None:
            return default
        return features.get(feature, default)
    
    df_spotify['valence'] = df_spotify['name'].apply(
        lambda x: get_feature(x, 'valence', 0.5)
    )
    df_spotify['arousal'] = df_spotify['name'].apply(
        lambda x: get_feature(x, 'arousal', 0.5)
    )
    df_spotify['tempo'] = df_spotify['name'].apply(
        lambda x: get_feature(x, 'tempo', 120.0)
    )
    
    lyrics_dict = {}
    if not skip_lyrics:
        logger.info("\n[3/6] Fetching lyrics from Genius...")
        try:
            lyrics_fetcher = LyricsFetcher(output_dir="lyrics")
            lyrics_paths = lyrics_fetcher.fetch_and_save_batch(tracks, delay=1.0, n_jobs=n_jobs)
            
            for track_name, filepath in lyrics_paths.items():
                if filepath:
                    # Handle both regular filepaths and 'CACHED:filepath' format
                    if filepath.startswith('CACHED:'):
                        filepath = filepath.replace('CACHED:', '')
                    lyrics_dict[track_name] = lyrics_fetcher.load_lyrics_from_file(filepath)
                else:
                    lyrics_dict[track_name] = None
            
            successful = sum(1 for v in lyrics_dict.values() if v)
            logger.info("  Fetched lyrics for %s/%s tracks", successful, len(tracks))
        except ValueError as e:
            logger.error("  Error: %s", e)
            logger.info("  Skipping lyrics (run with --skip-lyrics to avoid this)")
        except KeyboardInterrupt:
            logger.info("  Continuing with partial lyrics...")
            for track_name in [t['name'] for t in tracks]:
                if track_name not in lyrics_dict:
                    lyrics_dict[track_name] = None
        except Exception as e:
            logger.error("  Unexpected error: %s", e)
            logger.info("  Continuing without lyrics...")
    else:
        logger.info("\n[3/6] Skipping lyrics fetching...")
    
    logger.info("\n[4/6] Analyzing sentiment from lyrics...")
    if lyrics_dict:
        try:
            sentiment_analyzer = SentimentAnalyzer()
            df_sentiment = sentiment_analyzer.analyze_batch(lyrics_dict, n_jobs=n_jobs)
            logger.info("  Sentiment analysis complete")
            logger.info(
                "  Average sentiment: %.3f",
                df_sentiment["sentiment_compound"].mean(),
            )
        except Exception as e:
            logger.error("  Error: %s", e)
            df_sentiment = None
    else:
        logger.info("  No lyrics available")
        df_sentiment = None
    
    logger.info("\n[5/6] Combining mood data...")
    try:
        processor = MoodDataProcessor()
        
        if df_sentiment is None:
            df_sentiment = df_spotify[['name']].copy()
            df_sentiment['track_name'] = df_sentiment['name']
            df_sentiment['sentiment_compound'] = 0.0
            df_sentiment['sentiment_pos'] = 0.0
            df_sentiment['sentiment_neu'] = 1.0
            df_sentiment['sentiment_neg'] = 0.0
        
        df_mood = processor.combine_features(df_spotify, lyrics_dict, df_sentiment)
        
        df_mood = processor.normalize_valence_arousal(df_mood)
        
        mood_csv_path = output_path / "mood_data.csv"
        processor.save_mood_data(df_mood, mood_csv_path)
        
        stats = processor.generate_summary_stats(df_mood)
        
        logger.info("  Combined data created")
        logger.info("  Overall mood: %s", stats["overall_mood"])
        logger.info("  Avg valence: %.3f", stats["avg_valence"])
        logger.info("  Avg arousal: %.3f", stats["avg_arousal"])

    except Exception as e:
        logger.error("  Error: %s", e)
        return
    
    logger.info("\n[6/6] Creating visualizations...")
    try:
        visualizer = MoodVisualizer(output_dir=output_dir)
        
        visualizer.plot_mood_timeline(df_mood, save_path=output_path / "mood_timeline.png")
        logger.info("  Mood timeline created")
        
        visualizer.plot_valence_arousal_space(df_mood, save_path=output_path / "valence_arousal.png")
        logger.info("  Valence-arousal plot created")
        
        colors = processor.create_color_timeline(df_mood)
        visualizer.create_color_gradient(colors, save_path=output_path / "mood_gradient.png")
        logger.info("  Color gradient created")
        
        visualizer.plot_summary_dashboard(df_mood, stats, save_path=output_path / "dashboard.png")
        logger.info("  Summary dashboard created")
        
        try:
            visualizer.create_mood_video(
                df_mood, 
                colors, 
                duration_seconds=60,
                fps=30,
                save_path=output_path / "mood_video.mp4"
            )
            logger.info("  Mood video created")
        except ImportError as e:
            logger.warning("  Warning: Video creation skipped: %s", e)
        except Exception as e:
            logger.warning("  Warning: Video creation failed: %s", e)

    except Exception as e:
        logger.error("  Error: %s", e)

    logger.info("\nPIPELINE COMPLETE!")
    logger.info("All outputs saved to: %s", output_path.absolute())
    logger.info("\nGenerated files:")
    logger.info("  - mood_data.csv - Complete dataset")
    logger.info("  - mood_timeline.png - Valence/arousal/sentiment over time")
    logger.info("  - valence_arousal.png - 2D mood space visualization")
    logger.info("  - mood_gradient.png - Color gradient for diffusion models")
    logger.info("  - dashboard.png - Summary dashboard")
    logger.info("  - mood_video.mp4 - 180-second mood visualization video")
    logger.info("  - audio_cache/ - Downloaded audio files")
    logger.info("  - lyrics/ - Lyrics text files")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Spotify Mood Visualization Pipeline"
    )
    parser.add_argument(
        "playlist_url",
        type=str,
        help="Spotify playlist URL or URI"
    )
    parser.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Skip lyrics fetching and sentiment analysis"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio download and analysis (use default values)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh playlist data (ignore Spotify cache)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs/cores to use (-1 uses all cores, default: -1)"
    )
    
    args = parser.parse_args()
    
    main(
        args.playlist_url, 
        skip_lyrics=args.skip_lyrics,
        skip_audio=args.skip_audio,
        output_dir=args.output_dir,
        force_refresh=args.force_refresh,
        n_jobs=args.n_jobs
    )
