# Spotify Mood Visualization

![Representative frame from the generative mood video](docs/mood_video_preview.png)

A pipeline for analyzing a Spotify playlist and generating visuals based on
their audio and lyrics.

This project builds a track-by-track mood dataset using:
- **Valence** (musical positivity, 0.0 to 1.0)
- **Arousal** (energy/intensity, 0.0 to 1.0)
- **Sentiment** (lyrical polarity, -1.0 to 1.0)

It then renders charts and a generative mood video pipeline from that data.

## Quick Start

### 1) Install system dependency (`ffmpeg`)

macOS:
```bash
brew install ffmpeg
```

Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

Windows:
- Install from [ffmpeg.org](https://ffmpeg.org/download.html)

### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3) Create `.env`

```bash
# Required: Spotify
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Optional: Genius (lyrics sentiment)
GENIUS_CLIENT_ACCESS_TOKEN=your_genius_client_access_token
```
You'll need to create these using each platform's API website. 
- For Genius, use **Client Access Token** only (not Client ID/Secret).
- Lyrics analysis is optional. You can skip with `--skip-lyrics`.

### 4) Run

Always quote playlist URLs in `zsh`/`bash`:

```bash
python main.py "https://open.spotify.com/playlist/YOUR_PLAYLIST_ID"
```

Using optional command-line flags:
```bash
# no lyrics version
python main.py "YOUR_PLAYLIST_URL" --skip-lyrics

# test the core logic
python main.py "YOUR_PLAYLIST_URL" --skip-audio --skip-lyrics

# custom output directory
python main.py "YOUR_PLAYLIST_URL" --output-dir my_outputs
```

## The pipeline, in brief

1. Pulls playlist metadata from Spotify.
2. Downloads matching track audio from YouTube (`yt-dlp`).
3. Extracts audio features with `librosa`.
4. Computes valence/arousal from weighted feature heuristics.
5. Optionally fetches lyrics from Genius and scores sentiment with VADER.
6. Writes dataset and visualization outputs.

## Project Structure

```text
spotify_mood_visualisation/
├── main.py
├── requirements.txt
├── config_example.py
├── src/
│   ├── spotify_client.py
│   ├── audio_analyzer.py
│   ├── lyrics_fetcher.py
│   ├── sentiment_analyzer.py
│   ├── data_processor.py
│   └── visualizer.py
├── audio_cache/              # Downloaded audio cache (generated)
├── lyrics/                   # Lyrics cache (generated)
└── outputs/                  # CSVs, plots, video (generated)
```

## Outputs

You will find that `outputs/` is populated with:
- `mood_data.csv` - merged per-track dataset
- `mood_timeline.png` - mood progression over playlist order
- `valence_arousal.png` - 2D mood-space scatter
- `mood_gradient.png` - sequential color gradient
- `dashboard.png` - summary chart panel
- `mood_video.mp4` - generated animated visualization

## Mood Model

### Valence
- A measure of positivity.
- Computed from audio features related to brightness, harmonic ratio, and chroma features using librosa.
- Typical interpretation:
  - `0.0 to 0.3`: lower/negative affect
  - `0.3 to 0.7`: neutral/mixed
  - `0.7 to 1.0`: positive/uplifting

### Arousal
- A measure of intensity/energy.
- Computed from tempo, RMS loudness, zero crossing, percussive ratio, and rolloff using librosa.
- Typical interpretation:
  - `0.0 to 0.3`: calm
  - `0.3 to 0.7`: moderate
  - `0.7 to 1.0`: energetic/intense

### Sentiment
- Lyrical polarity from NLTK VADER. A rough dictionary-based approach to lyrical sentiment analysis.
- Range `-1.0` (negative) to `+1.0` (positive).
- If lyrics are skipped/unavailable, sentiment defaults to neutral.

## Video Interpretation

The video visualization reflects mood data by mapping valence to color hue (cool for low, warm for high), arousal to saturation (muted to vivid), and sentiment to brightness (dark to bright). Visual forms (shapes, motion, effects) shift based on these values—valence guides geometry, arousal drives movement intensity, and sentiment tweaks visual brightness. As the playlist progresses, the video evolves smoothly to represent track-by-track mood changes.

The shapes used in the visualization are based on these numbers, too. We gain more faces to the shape with higher valence and arousal. So happy, energetic music has lots of faces, and sad, calm music has few. 


## Caching

The project caches quite a lot of data to make reruns faster:
- Spotify metadata cache
- Downloaded audio in `audio_cache/`
- Lyrics in `lyrics/`
- Feature cache in `.analysis_cache/`

If caching should prove to be annoying, here's a couple helpful options:

```bash
# Force fresh Spotify metadata
python main.py "YOUR_PLAYLIST_URL" --force-refresh

# Clear caches
python utils/clear_cache.py --all
```

## AI Statement

I wrote the code used in this pipeline with help from the Cursor Auto and Claude Sonnet 4.5 models,
mostly back in December 2025 but with some tweaking in March 2026.
