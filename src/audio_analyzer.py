"""
Audio Analyzer
Extracts valence and arousal from audio files using librosa.
Downloads audio from YouTube and analyzes it for mood features.
"""

import librosa
import numpy as np
import yt_dlp
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, List
from multiprocessing import Pool, cpu_count, TimeoutError as MPTimeoutError
from functools import partial
import warnings
warnings.filterwarnings('ignore')
import signal

# scipy.signal.hann compatibility for librosa is a bit hacky, sorry
try:
    from scipy.signal import hann
except ImportError:
    try:
        from scipy.signal.windows import hann
        # Monkey-patch for librosa compatibility
        import scipy.signal
        scipy.signal.hann = hann
    except ImportError:
        pass

DURATION_TOLERANCE_SECONDS = 30
TRACK_PROCESS_TIMEOUT_SECONDS = 120
PARENT_STALL_TIMEOUT_SECONDS = 200
logger = logging.getLogger(__name__)

# populate if you know any problematic ids that are causing issues in this script
SKIP_YOUTUBE_VIDEO_IDS = {}

def _is_blocked_video_entry(entry: dict) -> bool:
    """Return True if an entry points to a blocked YouTube video ID."""
    if not isinstance(entry, dict):
        return False
    entry_id = str(entry.get('id') or '')
    webpage_url = str(entry.get('webpage_url') or '')
    url = str(entry.get('url') or '')
    combined = f"{entry_id} {webpage_url} {url}"
    return any(video_id in combined for video_id in SKIP_YOUTUBE_VIDEO_IDS)


class AudioAnalyzer:
    """Analyzes audio files to extract valence and arousal features."""
    
    def __init__(self, sample_rate: int = 22050, cache_dir: str = ".analysis_cache"):
        """
        Initialize audio analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
            cache_dir: Directory to store cached analysis results
        """
        self.sr = sample_rate
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.blocked_tracks_file = self.cache_dir / "blocked_tracks.json"
    
    def _get_cache_filename(self, audio_path: str) -> Path:
        """Get cache filename for an audio file. MD5 hash of audio path is used as cache key."""
        audio_file = Path(audio_path)
        if not audio_file.exists():
            cache_key = hashlib.md5(str(audio_path).encode()).hexdigest()
        else:
            stat = audio_file.stat()
            cache_key = hashlib.md5(
                f"{audio_path}_{stat.st_size}_{stat.st_mtime}".encode()
            ).hexdigest()
        
        return self.cache_dir / f"analysis_{cache_key}.json"

    @staticmethod
    def _normalize_track_key(artist_name: str, track_name: str) -> str:
        """Normalize track identity key for consistent blocklist matching.
        This is used to ensure that the same track is treated as the same track
        even if the artist name or track name is slightly different.
        """
        return f"{artist_name.strip().lower()}::{track_name.strip().lower()}"

    def _load_blocked_track_keys(self) -> set:
        """Load blocked track keys from persistent cache.
        This is used to store the track keys that have been blocked so that they are not processed again.
        """
        if not self.blocked_tracks_file.exists():
            return set()
        try:
            with open(self.blocked_tracks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(str(x) for x in data)
            return set()
        except Exception:
            return set()

    def _save_blocked_track_key(self, artist_name: str, track_name: str):
        """Persist a blocked track so all scripts skip it next run."""
        key = self._normalize_track_key(artist_name, track_name)
        blocked = self._load_blocked_track_keys()
        if key in blocked:
            return
        blocked.add(key)
        try:
            with open(self.blocked_tracks_file, "w", encoding="utf-8") as f:
                json.dump(sorted(blocked), f, indent=2)
        except Exception:
            pass

    def get_blocked_track_keys(self) -> set:
        """Public accessor for blocked track keys."""
        return self._load_blocked_track_keys()

    def is_track_blocked(self, artist_name: str, track_name: str) -> bool:
        """Check whether a track is blocked globally."""
        key = self._normalize_track_key(artist_name, track_name)
        return key in self._load_blocked_track_keys()
    
    def _load_from_cache(self, audio_path: str) -> Optional[Dict[str, float]]:
        """Load analysis results from cache if available."""
        cache_file = self._get_cache_filename(audio_path)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Verify it's for the same file
            if cached_data.get('audio_path') == str(Path(audio_path).absolute()):
                # Remove metadata, return just the features
                features = {k: v for k, v in cached_data.items() if k != 'audio_path'}
                return features
            else:
                return None
        except Exception:
            # Cache file corrupted or invalid
            return None
    
    def _save_to_cache(self, audio_path: str, features: Dict[str, float]):
        """Save analysis results to cache."""
        cache_file = self._get_cache_filename(audio_path)
        
        try:
            # Include audio path for verification
            cache_data = {
                'audio_path': str(Path(audio_path).absolute()),
                **features
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            # Cache save failed, but don't fail the analysis
            pass
    
    def analyze_audio_file(self, audio_path: str, use_cache: bool = True) -> Dict[str, float]:
        """
        Analyze an audio file and extract mood features.
        
        Args:
            audio_path: Path to audio file (mp3, wav, etc.)
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary with valence, arousal, and supporting features
        """
        # Check cache first
        if use_cache:
            cached_features = self._load_from_cache(audio_path)
            if cached_features:
                return cached_features
        
        # Load audio with error handling
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Failed to load audio file {audio_path}: {e}") from e
        
        # Extract features
        features = {}
        
        # Tempo and beat features (for arousal)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        features['beat_strength'] = float(librosa.beat.beat_track(y=y, sr=sr)[0])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_var'] = float(np.var(spectral_centroid))
        
        # Spectral rolloff (brightness)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate (noisiness/energy)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_var'] = float(np.var(zcr))
        
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_var'] = float(np.var(rms))
        
        # MFCC (timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_var'] = float(np.var(mfcc))
        
        # Harmonic/Percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmonic_mean'] = float(np.mean(np.abs(y_harmonic)))
        features['percussive_mean'] = float(np.mean(np.abs(y_percussive)))
        
        # Chroma features (harmony/mode)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_var'] = float(np.var(chroma))
        
        # Calculate valence and arousal
        features['valence'] = self._calculate_valence(features)
        features['arousal'] = self._calculate_arousal(features)
        
        # Save to cache
        self._save_to_cache(audio_path, features)
        
        return features
    
    def _calculate_valence(self, features: Dict[str, float]) -> float:
        """
        Calculate valence (positive/negative emotion) from audio features.
        
        Based on research:
        - Major mode, higher spectral centroid -> positive valence
        - Minor mode, lower spectral centroid -> negative valence
        - Harmonic content influences valence
        
        Args:
            features: Dictionary of extracted audio features
            
        Returns:
            Valence score (0.0 to 1.0)
        """
        # Normalize features to 0-1 range
        
        # Spectral centroid (brightness) - brighter sounds tend to be more positive
        # Typical range: 1000-4000 Hz
        centroid_norm = np.clip(
            (features['spectral_centroid_mean'] - 1000) / 3000,
            0, 1
        )
        
        # Harmonic content - more harmonic often means more positive
        harmonic_ratio = features['harmonic_mean'] / (
            features['harmonic_mean'] + features['percussive_mean'] + 1e-10
        )
        
        # Chroma variation - major modes tend to have different chroma patterns
        # This is a simplified heuristic
        chroma_score = np.clip(features['chroma_mean'] * 5, 0, 1)
        
        # Weighted combination
        valence = (
            0.4 * centroid_norm +      # Brightness
            0.3 * harmonic_ratio +      # Harmonic content
            0.3 * chroma_score          # Tonal content
        )
        
        return float(np.clip(valence, 0, 1))
    
    def _calculate_arousal(self, features: Dict[str, float]) -> float:
        """
        Calculate arousal (energy/intensity) from audio features.
        
        Based on research:
        - Fast tempo, high energy -> high arousal
        - Slow tempo, low energy -> low arousal
        - Loudness and noisiness contribute to arousal
        
        Args:
            features: Dictionary of extracted audio features
            
        Returns:
            Arousal score (0.0 to 1.0)
        """
        # Tempo normalization (typical range: 60-180 BPM)
        tempo_norm = np.clip((features['tempo'] - 60) / 120, 0, 1)
        
        # RMS energy (loudness)
        # Normalize assuming typical range 0-0.5
        rms_norm = np.clip(features['rms_mean'] * 2, 0, 1)
        
        # Zero crossing rate (noisiness/energy)
        # Normalize assuming typical range 0-0.2
        zcr_norm = np.clip(features['zcr_mean'] * 5, 0, 1)
        
        # Percussive content (rhythmic energy)
        percussive_ratio = features['percussive_mean'] / (
            features['harmonic_mean'] + features['percussive_mean'] + 1e-10
        )
        
        # Spectral rolloff (high frequency content indicates energy)
        rolloff_norm = np.clip(
            (features['spectral_rolloff_mean'] - 2000) / 6000,
            0, 1
        )
        
        # Weighted combination
        arousal = (
            0.3 * tempo_norm +          # Tempo
            0.25 * rms_norm +           # Loudness
            0.2 * zcr_norm +            # Noisiness
            0.15 * percussive_ratio +   # Rhythmic energy
            0.1 * rolloff_norm          # High frequency content
        )
        
        return float(np.clip(arousal, 0, 1))
    
    def find_best_matching_video(self, 
                                 entries: List[dict], 
                                 expected_duration: Optional[float]) -> Optional[dict]:
        """
        Find the best matching video from search results based on duration.
        
        Args:
            entries: List of video entries from yt-dlp search
            expected_duration: Expected duration in seconds (from Spotify)
            
        Returns:
            Best matching video dict, or None
        """
        if not entries:
            return None
        
        # Temporary hard-skip for known problematic video IDs.
        entries = [entry for entry in entries if not _is_blocked_video_entry(entry)]
        if not entries:
            return None
        
        if not expected_duration:
            # No duration to match against, return first valid entry
            return entries[0] if entries else None
        
        # Filter and sort by duration difference
        valid_entries = []
        for entry in entries:
            duration = entry.get('duration')
            if duration:
                diff = abs(duration - expected_duration)
                valid_entries.append((diff, entry))
        
        if not valid_entries:
            # No entries with duration info, return first entry
            return entries[0] if entries else None
        
        # Sort by duration difference (closest match first)
        valid_entries.sort(key=lambda x: x[0])
        
        best_diff, best_entry = valid_entries[0]
        
        # Check if within tolerance
        if best_diff <= DURATION_TOLERANCE_SECONDS:
            return best_entry
        
        # Best match is still too far off
        return None
    
    def download_and_analyze_from_youtube(self, 
                                          track_name: str, 
                                          artist_name: str,
                                          output_dir: str = "audio_cache",
                                          expected_duration_ms: Optional[int] = None,
                                          max_retries: int = 3,
                                          parallel_mode: bool = False) -> Optional[Dict[str, float]]:
        """
        Search for track on YouTube, download, and analyze.
        Uses duration matching to find the correct version.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            output_dir: Directory to save downloaded audio
            expected_duration_ms: Expected duration in milliseconds (from Spotify)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with valence, arousal, and features, or None if failed
        """
        import time
        import re

        # once a track is blocked, never query YouTube for it again
        track_key = self._normalize_track_key(artist_name, track_name)
        if track_key in self._load_blocked_track_keys():
            logger.warning("Skipping persistently blocked track: %s - %s", artist_name, track_name)
            return None
        
        # Convert duration to seconds
        expected_duration = expected_duration_ms / 1000 if expected_duration_ms else None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sanitize filename
        def sanitize_filename(text):
            # Remove invalid characters
            text = re.sub(r'[<>:"/\\|?*]', '', text)
            # Replace commas and other problematic chars
            text = text.replace(',', '_').replace('..', '_')
            # Limit length
            return text[:100]
        
        safe_artist = sanitize_filename(artist_name)
        safe_track = sanitize_filename(track_name)
        
        # Check if already downloaded - if exists, return None so batch processor handles it
        # (We don't analyze here - that happens in batch_download_and_analyze)
        audio_file = output_path / f"{safe_artist}_{safe_track}.mp3"
        if audio_file.exists():
            # File already exists - return None so caller knows to use cached file
            return None
        
        # Construct search query
        search_query = f"{artist_name} {track_name}"
        
        # Search more results when we have expected duration to filter
        search_count = 10 if expected_duration else 3
        
        # Configure yt-dlp with anti-detection measures
        import os as _os
        # In parallel mode each worker gets its own archive to avoid concurrent
        # writes to the same file corrupting it.
        archive_file = (
            str(output_path / f'downloaded_songs_{_os.getpid()}.txt')
            if parallel_mode
            else str(output_path / 'downloaded_songs.txt')
        )
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(output_path / f'{safe_artist}_{safe_track}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'default_search': f'ytsearch{search_count}',  # Search multiple results
            # Anti-detection options
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                }
            },
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'referer': 'https://www.youtube.com/',
            # skip sleep intervals in parallel mode
            'sleep_interval': 0 if parallel_mode else 1,
            'max_sleep_interval': 0 if parallel_mode else 3,
            'socket_timeout': 30,
            # per-worker archive to avoid race conditions in parallel mode
            'download_archive': archive_file,
        }
        
        # try with cookies if available
        cookie_file = Path.home() / '.config' / 'yt-dlp' / 'cookies.txt'
        if cookie_file.exists():
            ydl_opts['cookiefile'] = str(cookie_file)
        
        # Retry logic
        last_error = None
        for attempt in range(max_retries):
            try:
                # Add delay between retries (not before first attempt)
                if attempt > 0:
                    delay_seconds = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                    time.sleep(delay_seconds)
                    logger.info("Retry %s/%s...", attempt, max_retries - 1)
                
                # Search and filter by duration
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # First, extract info without downloading
                    search_results = ydl.extract_info(search_query, download=False)
                    
                    # Get entries
                    entries = []
                    if isinstance(search_results, dict) and search_results.get('entries'):
                        entries = search_results['entries']
                    elif isinstance(search_results, dict):
                        entries = [search_results]
                    elif isinstance(search_results, list):
                        entries = search_results
                    
                    # hard-skip blocked video IDs before any matching/retry path
                    entries = [entry for entry in entries if not _is_blocked_video_entry(entry)]
                    
                    if not entries:
                        logger.error("No videos found")
                        return None
                    
                    # Find best matching video by duration
                    best_video = self.find_best_matching_video(entries, expected_duration)
                    
                    if not best_video:
                        if expected_duration:
                            logger.error(
                                "No video within %ss of expected %.0fs",
                                DURATION_TOLERANCE_SECONDS,
                                expected_duration,
                            )
                        else:
                            logger.error("No valid video found")
                        return None
                    
                    # Check duration difference
                    video_duration = best_video.get('duration')
                    if expected_duration and video_duration:
                        diff = abs(video_duration - expected_duration)
                        if diff > 2:  # Log if noticeable difference
                            logger.info(
                                "Duration match: %.1fs difference (expected %.0fs, found %.0fs)",
                                diff,
                                expected_duration,
                                video_duration,
                            )
                    
                    # Download the best match
                    target = best_video.get('webpage_url') or best_video.get('url')
                    if not target:
                        logger.error("No URL for video")
                        return None
                    
                    # extra guard to stop blocked id if it somehow got here
                    if _is_blocked_video_entry(best_video):
                        logger.warning("Skipping blocked video ID: %s", best_video.get("id"))
                        return None
                    
                    info = ydl.extract_info(target, download=True)
                    
                # Download completed - now find the file
                downloaded_file = None
                if audio_file.exists():
                    downloaded_file = audio_file
                else:
                    # Try to find file with different extension
                    for ext in ['mp3', 'm4a', 'webm', 'opus']:
                        alt_file = output_path / f"{safe_artist}_{safe_track}.{ext}"
                        if alt_file.exists():
                            downloaded_file = alt_file
                            break
                
                if not downloaded_file:
                    # File not found after download - this is a download failure, will retry
                    last_error = f"Downloaded file not found: {audio_file}"
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error("%s", last_error)
                        return None
                
                # Download succeeded - now analyze (don't retry download if analysis fails)
                try:
                    features = self.analyze_audio_file(str(downloaded_file))
                    features['youtube_title'] = info.get('title', '')
                    features['youtube_duration'] = info.get('duration', 0)
                    features['audio_file'] = str(downloaded_file)
                    # SUCCESS - return immediately
                    return features
                except Exception as e:
                    # Analysis failed - don't retry download, just return None
                    logger.error("Analysis failed: %s", str(e)[:100])
                    return None
                    
            except yt_dlp.utils.DownloadError as e:
                error_str = str(e)
                last_error = error_str

                # temporary hard-skip for known problematic YouTube IDs 
                # even when the failure happens before/around selection logic
                if any(video_id in error_str for video_id in SKIP_YOUTUBE_VIDEO_IDS):
                    self._save_blocked_track_key(artist_name, track_name)
                    logger.warning(
                        "Skipping blocked video from download error: %s",
                        error_str[:100],
                    )
                    return None
                
                if '403' in error_str or 'Forbidden' in error_str:
                    if attempt < max_retries - 1:
                        # rotate through player clients on each retry to avoid being blocked by youtube
                        _client_rotation = [['web'], ['ios'], ['mweb'], ['web_creator']]
                        ydl_opts['extractor_args']['youtube']['player_client'] = (
                            _client_rotation[attempt % len(_client_rotation)]
                        )
                        continue
                    else:
                        logger.error("YouTube blocked download (403 Forbidden)")
                        logger.info("Tip: Update yt-dlp: pip install -U yt-dlp")
                        return None
                else:
                    # Other download errors - retry if attempts left
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error("Download failed: %s", error_str[:100])
                        return None
                    
            except Exception as e:
                last_error = str(e)
                # Other errors - retry if attempts left
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error("Error: %s", str(e)[:100])
                    return None
        
        # If we get here, all retries exhausted
        if last_error:
            logger.error("Failed after %s attempts: %s", max_retries, last_error[:100])
        return None
    
    @staticmethod
    def estimate_from_metadata(tempo: Optional[float] = None,
                               loudness: Optional[float] = None,
                               mode: Optional[int] = None) -> Dict[str, float]:
        """
        Rough estimation of valence/arousal from basic metadata.
        
        This is a fallback when audio analysis is not possible.
        Less accurate but better than nothing.
        
        Args:
            tempo: Beats per minute (60-180 typical)
            loudness: Loudness in dB (-60 to 0 typical)
            mode: Musical mode (0=minor, 1=major)
            
        Returns:
            Dictionary with estimated valence and arousal
        """
        # Default values
        valence = 0.5
        arousal = 0.5
        
        if tempo is not None:
            # Tempo influences arousal
            arousal = np.clip((tempo - 60) / 120, 0.2, 0.9)
        
        if mode is not None:
            # Major mode (1) tends to be more positive
            valence = 0.65 if mode == 1 else 0.35
        
        if loudness is not None:
            # Loudness affects both arousal and valence slightly
            loudness_norm = np.clip((loudness + 60) / 60, 0, 1)
            arousal = (arousal + loudness_norm) / 2
        
        return {
            'valence': float(valence),
            'arousal': float(arousal),
            'estimated': True  # Flag to indicate this is estimated
        }


    def batch_download_and_analyze(self, 
                                   tracks: list,
                                   output_dir: str = "audio_cache",
                                   delay: float = 2.0,
                                   use_duration_matching: bool = True,
                                   n_jobs: int = -1) -> Dict[str, Dict[str, float]]:
        """
        Download and analyze multiple tracks from YouTube (parallelized).
        
        Args:
            tracks: List of dicts with 'name', 'artist', and optionally 'duration_ms' keys
            output_dir: Directory to save downloaded audio
            delay: Delay between downloads (seconds) - not used in parallel mode
            use_duration_matching: Whether to filter results by duration
            n_jobs: Number of parallel jobs (-1 = all CPU cores)
            
        Returns:
            Dictionary mapping track names to feature dictionaries
        """
        import time
        import re
        from tqdm import tqdm
        
        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_jobs = max(1, min(n_jobs, len(tracks)))  # Don't exceed number of tracks
        
        def sanitize_filename(text):
            text = re.sub(r'[<>:"/\\|?*]', '', text)
            text = text.replace(',', '_').replace('..', '_')
            return text[:100]
        
        # Prepare arguments for worker function
        worker_args = [
            (track, output_dir, use_duration_matching, self.sr, str(self.cache_dir))
            for track in tracks
        ]
        
        # Process in parallel
        logger.info("Processing %s tracks with %s workers...", len(tracks), n_jobs)
        
        results_list = []
        with Pool(processes=n_jobs) as pool:
            # if no completed result arrives for too long, force-terminate stuck workers and 
            # continue pipeline with failures.
            iterator = pool.imap_unordered(_process_track_worker, worker_args, chunksize=1)
            progress = tqdm(total=len(tracks), desc="  Processing tracks")
            last_result_ts = time.time()
            try:
                while len(results_list) < len(tracks):
                    try:
                        worker_result = iterator.next(timeout=1.0)
                        results_list.append(worker_result)
                        progress.update(1)
                        last_result_ts = time.time()
                    except MPTimeoutError:
                        if time.time() - last_result_ts > PARENT_STALL_TIMEOUT_SECONDS:
                            logger.warning(
                                "No completed tracks for %ss; terminating stuck workers and skipping remaining tracks.",
                                PARENT_STALL_TIMEOUT_SECONDS,
                            )
                            pool.terminate()
                            break
            finally:
                progress.close()
        
        # if pool was terminated due to stall, mark unfinished tracks as failed
        if len(results_list) < len(tracks):
            completed_keys = {(name, artist) for name, artist, _, _ in results_list}
            for track in tracks:
                key = (track['name'], track['artist'])
                if key not in completed_keys:
                    duration_ms = track.get('duration_ms') if use_duration_matching else None
                    results_list.append((track['name'], track['artist'], duration_ms, None))
        
        # Convert list of results to dictionary
        results = {}
        successful = 0
        failed = 0
        
        for i, worker_result in enumerate(results_list, 1):
            track_name, artist_name, duration_ms, result = worker_result
            
            duration_str = f" ({duration_ms/1000:.0f}s)" if duration_ms else ""
            
            if result:
                results[track_name] = result
                successful += 1
                logger.info("[%s/%s] %s - %s%s", i, len(tracks), artist_name, track_name, duration_str)
                logger.info(
                    "Valence: %.3f, Arousal: %.3f",
                    result["valence"],
                    result["arousal"],
                )
            else:
                results[track_name] = None
                failed += 1
                logger.info("[%s/%s] %s - %s%s", i, len(tracks), artist_name, track_name, duration_str)
                logger.error("Failed")
        
        # sequential retry pass for tracks that failed in the parallel batch
        # this catches tracks that were killed by the parent watchdog etc.
        retry_tracks = [
            t for t in tracks
            if results.get(t['name']) is None
        ]
        if retry_tracks:
            logger.info("Retrying %s failed tracks sequentially...", len(retry_tracks))
            retry_analyzer = AudioAnalyzer(sample_rate=self.sr, cache_dir=str(self.cache_dir))
            for track in retry_tracks:
                t_name = track['name']
                t_artist = track['artist']
                t_duration = track.get('duration_ms') if use_duration_matching else None
                try:
                    retry_result = retry_analyzer.download_and_analyze_from_youtube(
                        t_name,
                        t_artist,
                        output_dir,
                        expected_duration_ms=t_duration,
                        parallel_mode=False,
                    )
                    if retry_result:
                        results[t_name] = retry_result
                        successful += 1
                        failed -= 1
                        logger.info(
                            "Retry success: %s - %s (valence=%.3f, arousal=%.3f)",
                            t_artist,
                            t_name,
                            retry_result["valence"],
                            retry_result["arousal"],
                        )
                    else:
                        logger.error("Retry failed: %s - %s", t_artist, t_name)
                except Exception as exc:
                    logger.error("Retry error: %s - %s: %s", t_artist, t_name, exc)

        logger.info("Summary: %s successful, %s failed", successful, failed)
        return results


def _process_track_worker(args):
    """
    Worker function for processing a single track (must be at module level for pickling).
    
    Args:
        args: Tuple of (track_dict, output_dir, use_duration_matching, sample_rate, cache_dir)
        
    Returns:
        Tuple of (track_name, artist_name, duration_ms, features_or_none)
    """
    track, output_dir, use_duration_matching, sample_rate, cache_dir = args
    
    import re
    from pathlib import Path
    
    # Create analyzer instance in worker process
    analyzer = AudioAnalyzer(sample_rate=sample_rate, cache_dir=cache_dir)
    
    track_name = track['name']
    artist_name = track['artist']
    duration_ms = track.get('duration_ms') if use_duration_matching else None

    def _timeout_handler(signum, frame):
        raise TimeoutError("Track processing timed out")
    
    def sanitize_filename(text):
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        text = text.replace(',', '_').replace('..', '_')
        return text[:100]
    
    safe_artist = sanitize_filename(artist_name)
    safe_track = sanitize_filename(track_name)
    audio_file = Path(output_dir) / f"{safe_artist}_{safe_track}.mp3"
    
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TRACK_PROCESS_TIMEOUT_SECONDS)

        if audio_file.exists():
            try:
                features = analyzer.analyze_audio_file(str(audio_file), use_cache=True)
                features['audio_file'] = str(audio_file)
                return (track_name, artist_name, duration_ms, features)
            except Exception:
                return (track_name, artist_name, duration_ms, None)
        else:
            try:
                download_result = analyzer.download_and_analyze_from_youtube(
                    track_name,
                    artist_name,
                    output_dir,
                    expected_duration_ms=duration_ms,
                    parallel_mode=True,
                )
                
                if download_result:
                    return (track_name, artist_name, duration_ms, download_result)
                
                if audio_file.exists():
                    try:
                        features = analyzer.analyze_audio_file(str(audio_file))
                        features['audio_file'] = str(audio_file)
                        return (track_name, artist_name, duration_ms, features)
                    except Exception:
                        return (track_name, artist_name, duration_ms, None)
                
                for ext in ['mp3', 'm4a', 'webm', 'opus']:
                    alt_file = Path(output_dir) / f"{safe_artist}_{safe_track}.{ext}"
                    if alt_file.exists():
                        try:
                            features = analyzer.analyze_audio_file(str(alt_file))
                            features['audio_file'] = str(alt_file)
                            return (track_name, artist_name, duration_ms, features)
                        except Exception:
                            continue
                
                return (track_name, artist_name, duration_ms, None)
            except Exception:
                return (track_name, artist_name, duration_ms, None)
    except TimeoutError:
        logger.warning("Track timeout skipped: %s - %s", artist_name, track_name)
        return (track_name, artist_name, duration_ms, None)
    finally:
        signal.alarm(0)



