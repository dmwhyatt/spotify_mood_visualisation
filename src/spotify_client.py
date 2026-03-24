"""
Spotify API Client
Handles authentication and fetching playlist tracks with audio features.
"""

import logging
import os
import json
import hashlib
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SpotifyClient:
    """Client for interacting with Spotify API."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, cache_dir: str = ".spotify_cache"):
        """
        Initialize Spotify client.
        
        Args:
            client_id: Spotify client ID (defaults to SPOTIFY_CLIENT_ID env var)
            client_secret: Spotify client secret (defaults to SPOTIFY_CLIENT_SECRET env var)
            cache_dir: Directory to store cached playlist data
        """
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")

        auth_manager = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_playlist_tracks(self, playlist_url: str, use_cache: bool = True) -> List[Dict]:
        """
        Get all tracks from a Spotify playlist.
        
        Args:
            playlist_url: Spotify playlist URL or URI
            use_cache: Whether to use cached data if available
            
        Returns:
            List of track dictionaries with metadata
        """
        playlist_id = self._extract_playlist_id(playlist_url)
        
        if use_cache:
            cached_tracks = self._load_from_cache(playlist_id)
            if cached_tracks:
                logger.info("Using cached playlist data (%s tracks)", len(cached_tracks))
                return cached_tracks
        
        tracks = []
        results = self.sp.playlist_tracks(playlist_id)
        
        while results:
            for item in results['items']:
                if item['track']:
                    track = item['track']
                    tracks.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artist': ', '.join([artist['name'] for artist in track['artists']]),
                        'album': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'position': len(tracks) + 1
                    })
            
            results = self.sp.next(results) if results['next'] else None
        
        self._save_to_cache(playlist_id, tracks)
        
        return tracks
    
    def get_playlist_tracks_df(self, playlist_url: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Get playlist tracks as a DataFrame.
        
        Args:
            playlist_url: Spotify playlist URL or URI
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with track metadata
        """
        tracks = self.get_playlist_tracks(playlist_url, use_cache=use_cache)
        return pd.DataFrame(tracks)
    
    def _extract_playlist_id(self, playlist_url: str) -> str:
        """Extract playlist ID from URL or URI."""
        if 'spotify:playlist:' in playlist_url:
            return playlist_url.split('spotify:playlist:')[-1]
        elif 'open.spotify.com/playlist/' in playlist_url:
            return playlist_url.split('open.spotify.com/playlist/')[-1].split('?')[0]
        else:
            return playlist_url
    
    def _get_cache_filename(self, playlist_id: str) -> Path:
        """Get cache filename for a playlist."""
        id_hash = hashlib.md5(playlist_id.encode()).hexdigest()[:8]
        return self.cache_dir / f"playlist_{playlist_id}_{id_hash}.json"
    
    def _save_to_cache(self, playlist_id: str, tracks: List[Dict]):
        """Save playlist tracks to cache."""
        cache_file = self._get_cache_filename(playlist_id)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'playlist_id': playlist_id,
                    'track_count': len(tracks),
                    'tracks': tracks
                }, f, indent=2)
        except Exception as e:
            logger.warning("Could not save cache: %s", e)
    
    def _load_from_cache(self, playlist_id: str) -> Optional[List[Dict]]:
        """Load playlist tracks from cache."""
        cache_file = self._get_cache_filename(playlist_id)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get('playlist_id') == playlist_id:
                return data.get('tracks', [])
            else:
                return None
        except Exception as e:
            logger.warning("Could not load cache: %s", e)
            return None
    
    def clear_cache(self, playlist_id: Optional[str] = None):
        """
        Clear cached playlist data.
        
        Args:
            playlist_id: Specific playlist to clear, or None to clear all
        """
        if playlist_id:
            cache_file = self._get_cache_filename(playlist_id)
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Cleared cache for playlist %s", playlist_id)
        else:
            for cache_file in self.cache_dir.glob("playlist_*.json"):
                cache_file.unlink()
            logger.info("Cleared all playlist caches")
