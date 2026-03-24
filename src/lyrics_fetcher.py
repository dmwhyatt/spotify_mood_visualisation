"""
Lyrics Fetcher
Fetches lyrics for tracks and saves them to text files using the Genius API.
"""

import os
import re
import requests
from typing import Optional, List
from pathlib import Path
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class LyricsFetcher:
    """Class to grab lyrics directly from Genius API."""
    
    def __init__(self, genius_token: Optional[str] = None, output_dir: str = "lyrics"):
        """
        Initialize lyrics fetcher with direct API access.
        
        Args:
            genius_token: Genius API access token (Client Access Token)
            output_dir: Directory to save lyrics text files
        """
        self.genius_token = (
            genius_token or 
            os.getenv('GENIUS_CLIENT_ACCESS_TOKEN') or
            os.getenv('GENIUS_ACCESS_TOKEN') or
            os.getenv('GENIUS_CLIENT_SECRET')
        )
        
        if not self.genius_token:
            raise ValueError(
                "Genius access token not found.\n"
                "Get it from: https://genius.com/api-clients\n"
                "After creating an API client, copy the 'Client Access Token'\n"
                "Add to .env file: GENIUS_CLIENT_ACCESS_TOKEN=your_token_here\n"
                "Or skip lyrics with: --skip-lyrics"
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.genius_token}',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        
        self.base_url = 'https://api.genius.com'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def search_song(self, track_name: str, artist_name: str) -> Optional[dict]:
        """
        Search for a song on Genius.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            
        Returns:
            Song info dict with 'url' and 'title', or None if not found
        """
        try:
            search_url = f'{self.base_url}/search'
            params = {'q': f'{artist_name} {track_name}'}
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            json_data = response.json()
            hits = json_data.get('response', {}).get('hits', [])
            
            if not hits:
                return None

            result = hits[0].get('result', {})
            return {
                'url': result.get('url'),
                'title': result.get('title'),
                'artist': result.get('primary_artist', {}).get('name'),
                'id': result.get('id')
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                if not hasattr(self, '_warned_401'):
                    logger.warning(
                        "Genius API 401 - invalid token. Check GENIUS_CLIENT_ACCESS_TOKEN in .env"
                    )
                    self._warned_401 = True
            return None
        except Exception as e:
            return None
    
    def scrape_lyrics_from_url(self, url: str) -> Optional[str]:
        """
        Scrape lyrics from a Genius song URL using BeautifulSoup.
        
        Args:
            url: Genius song URL
            
        Returns:
            Lyrics text or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            lyrics_divs = (
                soup.find_all('div', {'data-lyrics-container': 'true'}) or
                soup.find_all('div', class_=re.compile(r'Lyrics__Container')) or
                []
            )
            
            if not lyrics_divs:
                return None
            
            lyrics_parts = []
            for div in lyrics_divs:
                lyrics_parts.append(div.get_text(separator='\n', strip=True))
            
            lyrics = '\n\n'.join(lyrics_parts)
            
            lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
            lyrics = lyrics.strip()
            
            return lyrics if lyrics else None
            
        except Exception as e:
            return None
    
    def fetch_lyrics(self, track_name: str, artist_name: str) -> Optional[str]:
        """
        Fetch lyrics for a track.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            
        Returns:
            Lyrics text or None if not found
        """
        song_info = self.search_song(track_name, artist_name)
        
        if not song_info or not song_info.get('url'):
            return None
        
        lyrics = self.scrape_lyrics_from_url(song_info['url'])
        
        return lyrics
    
    def save_lyrics(self, track_name: str, artist_name: str, lyrics: str, position: int) -> str:
        """
        Save lyrics to a text file.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            lyrics: Lyrics text
            position: Track position in playlist
            
        Returns:
            Path to saved file
        """
        safe_name = self._sanitize_filename(f"{position:03d}_{artist_name}_{track_name}")
        filepath = self.output_dir / f"{safe_name}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Track: {track_name}\n")
            f.write(f"Artist: {artist_name}\n")
            f.write(f"Position: {position}\n")
            f.write("-" * 50 + "\n\n")
            f.write(lyrics)
        
        return str(filepath)
    
    def is_lyrics_cached(self, position: int) -> Optional[str]:
        """
        Check if lyrics are already cached for a track.
        
        Args:
            position: Track position in playlist
            
        Returns:
            Path to cached lyrics file if exists and has content, None otherwise
        """
        pattern = f"{position:03d}_*.txt"
        existing_files = list(self.output_dir.glob(pattern))
        
        if not existing_files:
            return None
        
        filepath = existing_files[0]
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if '---' in content or '-' * 50 in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('-' * 50) or line.startswith('---'):
                        remaining = '\n'.join(lines[i+2:]).strip()
                        if len(remaining) > 10:  # Has actual lyrics
                            return str(filepath)
            
            # If no separator but has substantial content, assume it's valid
            if len(content.strip()) > 50:
                return str(filepath)
                
        except Exception:
            return None
        
        return None
    
    def fetch_and_save_batch(self, tracks: List[dict], delay: float = 0.5, n_jobs: int = -1) -> dict:
        """
        Fetch and save lyrics for a batch of tracks (parallelized).
        Skips tracks that already have cached lyrics.
        
        Args:
            tracks: List of track dictionaries with 'name', 'artist', and 'position'
            delay: Delay between requests (not used in parallel mode)
            n_jobs: Number of parallel jobs (-1 = all CPU cores)
            
        Returns:
            Dictionary mapping track names to lyrics file paths
        """
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_jobs = max(1, min(n_jobs, len(tracks)))
        
        genius_token = self.genius_token
        output_dir = str(self.output_dir)
        base_url = self.base_url
        
        worker_args = [
            (track, genius_token, output_dir, base_url)
            for track in tracks
        ]
        
        logger.info("Processing %s tracks with %s workers...", len(tracks), n_jobs)
        
        with Pool(processes=n_jobs) as pool:
            results_list = list(tqdm(
                pool.imap(_fetch_lyrics_worker, worker_args),
                total=len(tracks),
                desc="  Fetching lyrics"
            ))
        
        results = {}
        successful = 0
        failed = 0
        cached = 0
        
        for i, (track, result) in enumerate(zip(tracks, results_list), 1):
            track_name = track['name']
            artist_name = track['artist']
            
            if result and result != 'NOT_FOUND':
                results[track_name] = result
                successful += 1
                if result.startswith('CACHED:'):
                    cached += 1
                    logger.info("[%s/%s] Using cached: %s", i, len(tracks), track_name)
                else:
                    logger.info("[%s/%s] %s - %s", i, len(tracks), track_name, artist_name)
                    logger.info("Saved")
            else:
                results[track_name] = None
                failed += 1
                logger.info("[%s/%s] %s - %s", i, len(tracks), track_name, artist_name)
                logger.info("Not found")

        logger.info(
            "Summary: %s total (%s cached, %s new), %s failed",
            successful,
            cached,
            successful - cached,
            failed,
        )
        return results
    
    def load_lyrics_from_file(self, filepath: str) -> str:
        """
        Load lyrics from a saved text file.
        
        Args:
            filepath: Path to lyrics file
            
        Returns:
            Lyrics text content
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            lyrics_start = 0
            
            for i, line in enumerate(lines):
                if line.startswith('-' * 50) or line.startswith('---'):
                    lyrics_start = i + 2
                    break
            
            lyrics = '\n'.join(lines[lyrics_start:]).strip()
            return lyrics if lyrics else ""
            
        except Exception as e:
            logger.warning("Could not load lyrics from %s: %s", filepath, e)
            return ""
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Remove invalid characters from filename."""
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.replace(' ', '_')
        return filename[:200]


def _fetch_lyrics_worker(args):
    """
    Worker function for fetching lyrics (must be at module level for pickling).
    
    Args:
        args: Tuple of (track_dict, genius_token, output_dir, base_url)
        
    Returns:
        Filepath string, 'CACHED:filepath', or 'NOT_FOUND'
    """
    track, genius_token, output_dir, base_url = args
    
    track_name = track['name']
    artist_name = track['artist']
    position = track['position']
    
    lyrics_dir = Path(output_dir)
    pattern = f"{position:03d}_*.txt"
    existing_files = list(lyrics_dir.glob(pattern))
    
    if existing_files:
        filepath = existing_files[0]
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content.strip()) > 50:
                    return f"CACHED:{filepath}"
            except Exception:
                pass
    
    try:
        session = requests.Session()
        session.headers.update({
            'Authorization': f'Bearer {genius_token}',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        
        search_url = f'{base_url}/search'
        params = {'q': f'{artist_name} {track_name}'}
        response = session.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        
        json_data = response.json()
        hits = json_data.get('response', {}).get('hits', [])
        
        if not hits:
            return 'NOT_FOUND'
        
        result = hits[0].get('result', {})
        song_url = result.get('url')
        
        if not song_url:
            return 'NOT_FOUND'
        
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        lyrics_response = requests.get(song_url, headers=headers, timeout=10)
        lyrics_response.raise_for_status()
        
        soup = BeautifulSoup(lyrics_response.text, 'html.parser')
        lyrics_divs = (
            soup.find_all('div', {'data-lyrics-container': 'true'}) or
            soup.find_all('div', class_=re.compile(r'Lyrics__Container')) or
            []
        )
        
        if not lyrics_divs:
            return 'NOT_FOUND'
        
        lyrics_parts = []
        for div in lyrics_divs:
            lyrics_parts.append(div.get_text(separator='\n', strip=True))
        
        lyrics = '\n\n'.join(lyrics_parts)
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics).strip()
        
        if not lyrics:
            return 'NOT_FOUND'

        safe_name = re.sub(r'[<>:"/\\|?*]', '', f"{position:03d}_{artist_name}_{track_name}")
        safe_name = safe_name.replace(' ', '_')[:200]
        filepath = lyrics_dir / f"{safe_name}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Track: {track_name}\n")
            f.write(f"Artist: {artist_name}\n")
            f.write(f"Position: {position}\n")
            f.write("-" * 50 + "\n\n")
            f.write(lyrics)
        
        return str(filepath)
        
    except Exception:
        return 'NOT_FOUND'
