#!/usr/bin/env python3
"""
Utility script to manage caches.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spotify_client import SpotifyClient

logger = logging.getLogger(__name__)


def clear_spotify_cache(playlist_id: str = None):
    """Clear Spotify playlist cache."""
    client = SpotifyClient()
    client.clear_cache(playlist_id)


def clear_audio_cache():
    """Clear downloaded audio files."""
    audio_dir = Path("audio_cache")
    if audio_dir.exists():
        count = 0
        for file in audio_dir.glob("*"):
            if file.is_file():
                file.unlink()
                count += 1
        logger.info("Cleared %s audio files", count)
    else:
        logger.info("No audio cache found")


def clear_lyrics_cache():
    """Clear downloaded lyrics files."""
    lyrics_dir = Path("lyrics")
    if lyrics_dir.exists():
        count = 0
        for file in lyrics_dir.glob("*.txt"):
            file.unlink()
            count += 1
        logger.info("Cleared %s lyrics files", count)
    else:
        logger.info("No lyrics cache found")


def clear_analysis_cache():
    """Clear analysis results cache."""
    analysis_dir = Path(".analysis_cache")
    if analysis_dir.exists():
        count = 0
        for file in analysis_dir.glob("analysis_*.json"):
            file.unlink()
            count += 1
        logger.info("Cleared %s analysis cache files", count)
    else:
        logger.info("No analysis cache found")


def clear_all():
    """Clear all caches."""
    logger.info("Clearing all caches...")
    clear_spotify_cache()
    clear_audio_cache()
    clear_lyrics_cache()
    clear_analysis_cache()
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Clear caches")
    parser.add_argument(
        "--spotify",
        action="store_true",
        help="Clear Spotify playlist cache"
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Clear audio cache"
    )
    parser.add_argument(
        "--lyrics",
        action="store_true",
        help="Clear lyrics cache"
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Clear analysis results cache"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all caches"
    )
    parser.add_argument(
        "--playlist-id",
        type=str,
        help="Specific playlist ID to clear (for Spotify cache)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        clear_all()
    elif args.spotify:
        clear_spotify_cache(args.playlist_id)
    elif args.audio:
        clear_audio_cache()
    elif args.lyrics:
        clear_lyrics_cache()
    elif args.analysis:
        clear_analysis_cache()
    else:
        logger.warning("No cache specified. Use --help for options.")
        logger.info("")
        logger.info("Available caches:")
        logger.info("  --spotify  : Playlist metadata")
        logger.info("  --audio    : Downloaded audio files")
        logger.info("  --lyrics   : Downloaded lyrics files")
        logger.info("  --analysis : Analysis results (valence/arousal)")
        logger.info("  --all      : All of the above")

