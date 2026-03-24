"""
Script for creating simple (but hopefully interesting) visualizations from mood data.
Produces:
- Timeline plot
- Valence-arousal scatter plot
- Color gradient
- Summary dashboard
- Mood video
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


class MoodVisualizer:
    """Creates visualizations from mood data."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualizer. This class uses lyric sentiment and and audio features to create visualizations
        based on each track to construct a mood timeline, valence-arousal space, and color gradient. It also
        creates a mood video, which is a simple rule-based generative video that is created from the color gradient.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_mood_timeline(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot valence, arousal, and sentiment over the playlist timeline.
        
        Args:
            df: DataFrame with mood data
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(df['position'], df['valence'], 'o-', color='#FF6B6B', linewidth=2)
        axes[0].fill_between(df['position'], df['valence'], alpha=0.3, color='#FF6B6B')
        axes[0].set_ylabel('Valence (normalized)', fontsize=12)
        axes[0].set_title('Mood Timeline Across Playlist', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        axes[1].plot(df['position'], df['arousal'], 'o-', color='#4ECDC4', linewidth=2)
        axes[1].fill_between(df['position'], df['arousal'], alpha=0.3, color='#4ECDC4')
        axes[1].set_ylabel('Arousal (normalized)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        axes[2].plot(df['position'], df['sentiment_compound'], 'o-', color='#95E1D3', linewidth=2)
        axes[2].fill_between(df['position'], df['sentiment_compound'], alpha=0.3, color='#95E1D3')
        axes[2].set_ylabel('Sentiment', fontsize=12)
        axes[2].set_xlabel('Track Position', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-1, 1)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Timeline plot saved to %s", save_path)
        else:
            plt.savefig(self.output_dir / "mood_timeline.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_valence_arousal_space(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot tracks in 2D valence-arousal space (Russell's Circumplex Model).
        
        Args:
            df: DataFrame with mood data
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            df['valence'],
            df['arousal'],
            c=df['sentiment_compound'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sentiment', fontsize=12)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        ax.text(0.85, 0.85, 'Happy\nExcited', ha='center', va='center',
                fontsize=12, alpha=0.5, fontweight='bold')
        ax.text(0.15, 0.85, 'Angry\nTense', ha='center', va='center',
                fontsize=12, alpha=0.5, fontweight='bold')
        ax.text(0.15, 0.15, 'Sad\nDepressed', ha='center', va='center',
                fontsize=12, alpha=0.5, fontweight='bold')
        ax.text(0.85, 0.15, 'Calm\nRelaxed', ha='center', va='center',
                fontsize=12, alpha=0.5, fontweight='bold')
        
        ax.set_xlabel('Valence (normalized, Negative to Positive)', fontsize=12)
        ax.set_ylabel('Arousal (normalized, Calm to Energetic)', fontsize=12)
        ax.set_title('Valence-Arousal Space (Russell\'s Circumplex Model)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Valence-arousal plot saved to %s", save_path)
        else:
            plt.savefig(self.output_dir / "valence_arousal_space.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_color_gradient(self, colors: List[tuple], 
                             width: int = 1920, 
                             height: int = 1088,
                             save_path: Optional[str] = None) -> Image.Image:
        """
        Create a smooth color gradient image from the color timeline.
        
        This can be used as input for diffusion models.
        
        Args:
            colors: List of RGB tuples
            width: Image width
            height: Image height
            save_path: Optional path to save the image
            
        Returns:
            PIL Image
        """
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        n_colors = len(colors)
        if n_colors == 0:
            return Image.fromarray(img_array)
        
        section_width = width / n_colors
        
        for i in range(n_colors):
            x_start = int(i * section_width)
            x_end = int((i + 1) * section_width) if i < n_colors - 1 else width
            
            if i < n_colors - 1:
                color1 = np.array(colors[i])
                color2 = np.array(colors[i + 1])
                
                for x in range(x_start, x_end):
                    t = (x - x_start) / (x_end - x_start)
                    color = color1 * (1 - t) + color2 * t
                    img_array[:, x] = color.astype(np.uint8)
            else:
                img_array[:, x_start:x_end] = colors[i]
        
        img = Image.fromarray(img_array)
        
        if save_path:
            img.save(save_path)
            logger.info("Color gradient saved to %s", save_path)
        else:
            img.save(self.output_dir / "mood_gradient.png")
        
        return img
    
    def plot_summary_dashboard(self, df: pd.DataFrame, stats: dict, 
                               save_path: Optional[str] = None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            df: DataFrame with mood data
            stats: Dictionary with summary statistics
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['position'], df['valence'], 'o-', label='Valence (normalized)', linewidth=2)
        ax1.plot(df['position'], df['arousal'], 's-', label='Arousal (normalized)', linewidth=2)
        normalized_sentiment = (df['sentiment_compound'] + 1) / 2
        ax1.plot(df['position'], normalized_sentiment, '^-', label='Sentiment (normalized)', linewidth=2)
        ax1.set_xlabel('Track Position')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_title('Mood Metrics Timeline', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        scatter = ax2.scatter(df['valence'], df['arousal'], 
                             c=df['sentiment_compound'], cmap='RdYlGn',
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Valence (normalized)')
        ax2.set_ylabel('Arousal (normalized)')
        ax2.set_title('Valence-Arousal Space', fontweight='bold')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(df['valence'], bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')
        ax3.set_xlabel('Valence (normalized)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Valence Distribution', fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.axvline(stats['avg_valence'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax3.axvline(0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Midpoint')
        ax3.legend()
        
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(df['arousal'], bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax4.set_xlabel('Arousal (normalized)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Arousal Distribution', fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.axvline(stats['avg_arousal'], color='blue', linestyle='--', linewidth=2, label='Mean')
        ax4.axvline(0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Midpoint')
        ax4.legend()

        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""
        PLAYLIST SUMMARY STATISTICS
        
        Total Tracks: {stats['total_tracks']}
        Tracks with Lyrics: {stats['tracks_with_lyrics']}
        
        Average Valence (normalized): {stats['avg_valence']:.3f} ± {stats['valence_std']:.3f}
        Average Arousal (normalized): {stats['avg_arousal']:.3f} ± {stats['arousal_std']:.3f}
        Average Sentiment: {stats['avg_sentiment']:.3f} ± {stats['sentiment_std']:.3f}
        
        Note: Valence and Arousal are normalized to 0-1 range
        Overall Mood: {stats['overall_mood']}
        """
        
        ax5.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Spotify Playlist Mood Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Dashboard saved to %s", save_path)
        else:
            plt.savefig(self.output_dir / "mood_dashboard.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _create_voronoi_pattern_fast(self, width, height, seed_points, colors):
        """Create a Voronoi diagram pattern using vectorized operations."""
        scale = 2
        w_small, h_small = width // scale, height // scale

        y_coords, x_coords = np.ogrid[0:h_small, 0:w_small]

        seed_array = np.array(seed_points)
        seed_array_scaled = seed_array / scale

        distances = np.sqrt(
            (x_coords[None, :, :] - seed_array_scaled[:, 0, None, None])**2 +
            (y_coords[None, :, :] - seed_array_scaled[:, 1, None, None])**2
        )

        nearest = np.argmin(distances, axis=0)

        pattern_small = np.zeros((h_small, w_small, 3), dtype=np.uint8)
        for i in range(len(colors)):
            mask = nearest == i
            pattern_small[mask] = colors[i]

        pattern = cv2.resize(pattern_small, (width, height), interpolation=cv2.INTER_LINEAR)
        return pattern
    
    def _draw_particles_vectorized(self, frame, particles, color, size):
        """Draw particles efficiently using vectorized operations."""
        height, width = frame.shape[:2]

        valid = (particles[:, 0] >= size) & (particles[:, 0] < width - size) & \
                (particles[:, 1] >= size) & (particles[:, 1] < height - size)
        valid_particles = particles[valid].astype(int)

        for x, y in valid_particles:
            frame[y-size:y+size+1, x-size:x+size+1] = np.clip(
                frame[y-size:y+size+1, x-size:x+size+1] + color, 0, 255
            )

    def create_mood_video(self, 
                         df: pd.DataFrame, 
                         colors: List[tuple],
                         duration_seconds: int = 60,
                         fps: int = 30,
                         width: int = 1920,
                         height: int = 1088,
                         save_path: Optional[str] = None):
        """
        Create a visually striking generative video with geometric patterns
        that morphs through the playlist timeline using colors and mood dimensions.
        
        Optimized version using OpenCV and vectorized operations for performance. Not every aspect
        will update every frame, please forgive magic numbers.
        
        Features:
        - Rotating polygons and sacred geometry
        - Multi-layered particle systems
        - Voronoi diagrams
        - Fractal-like patterns
        - Dynamic color morphing based on valence/arousal
        
        Args:
            df: DataFrame with mood data (must have 'valence', 'arousal', 'position')
            colors: List of RGB tuples corresponding to each track
            duration_seconds: Video duration in seconds (default: 60)
            fps: Frames per second (default: 30)
            width: Video width in pixels (default: 1920)
            height: Video height in pixels (default: 1080)
            save_path: Optional path to save the video
        """
        if not HAS_IMAGEIO:
            raise ImportError("imageio is required for video creation. Install with: pip install imageio imageio-ffmpeg")
        
        if not HAS_TQDM:
            raise ImportError("tqdm is required for video creation. Install with: pip install tqdm")
        
        if not HAS_CV2:
            raise ImportError("opencv-python is required for optimized video rendering. Install with: pip install opencv-python")
        
        total_frames = duration_seconds * fps
        n_tracks = len(df)
        
        if n_tracks == 0:
            raise ValueError("DataFrame is empty")

        positions = df['position'].values
        valences = df['valence'].values
        arousals = df['arousal'].values

        if positions.max() > positions.min():
            normalized_positions = (positions - positions.min()) / (positions.max() - positions.min())
        else:
            normalized_positions = np.linspace(0, 1, n_tracks)

        timeline_positions = np.linspace(0, 1, total_frames)

        logger.info(
            "Creating %ss generative video at %sfps (%s frames)...",
            duration_seconds,
            fps,
            total_frames,
        )
        logger.info("Video dimensions: %sx%s", width, height)
        logger.info("Optimized rendering with OpenCV and vectorized operations")

        frames = []
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) * 0.45

        n_particles_flow = 200
        n_particles_orbit = 100
        n_particles_radial = 80

        particles_flow = np.random.rand(n_particles_flow, 2) * np.array([width, height])
        constant_speed = 4.0
        random_directions = np.random.rand(n_particles_flow, 2) - 0.5
        velocities_flow = (random_directions / (np.linalg.norm(random_directions, axis=1, keepdims=True) + 1e-10)) * constant_speed
        
        orbit_angles = np.random.rand(n_particles_orbit) * 2 * np.pi
        orbit_radii = np.random.rand(n_particles_orbit) * max_radius * 0.8
        
        radial_angles = np.random.rand(n_particles_radial) * 2 * np.pi
        radial_speeds = np.random.rand(n_particles_radial) * 3 + 1
        radial_phases = np.random.rand(n_particles_radial) * max_radius

        n_voronoi = 10
        voronoi_points = np.random.rand(n_voronoi, 2) * np.array([width, height])
        voronoi_velocities = (np.random.rand(n_voronoi, 2) - 0.5) * 0.5

        y_grad = np.linspace(0, 1, height).reshape(-1, 1)
        x_grad = np.linspace(0, 1, width).reshape(1, -1)
        y_coords = np.linspace(-1, 1, height).reshape(-1, 1)
        x_coords = np.linspace(-1, 1, width).reshape(1, -1)
        radial_grad = np.clip(np.sqrt(x_coords**2 + y_coords**2), 0, 1)
        bg_intensity = (y_grad * 0.3 + x_grad * 0.2 + (1 - radial_grad) * 0.5) * 0.15
        
        particle_history = []
        voronoi_cache = None
        voronoi_cache_counter = 0

        smoothed_color = np.array(colors[0] if colors else [128, 128, 128], dtype=float)
        color_smoothing_factor = 0.08

        initial_valence = valences[0] if len(valences) > 0 else 0.5
        initial_arousal = arousals[0] if len(arousals) > 0 else 0.5
        smoothed_focal_x = center_x + (initial_valence - 0.5) * max_radius * 1.4
        smoothed_focal_y = center_y - (initial_arousal - 0.5) * max_radius * 1.4
        focal_speed = 2.5

        for frame_idx, t in enumerate(tqdm(timeline_positions, desc="Generating frames", ncols=80)):
            track_idx = np.searchsorted(normalized_positions, t, side='right') - 1
            track_idx = np.clip(track_idx, 0, n_tracks - 1)
            
            if track_idx < n_tracks - 1:
                t_local = (t - normalized_positions[track_idx]) / (
                    normalized_positions[track_idx + 1] - normalized_positions[track_idx] + 1e-10
                )
                t_local = np.clip(t_local, 0, 1)
            else:
                t_local = 0

            if track_idx < n_tracks - 1:
                valence = valences[track_idx] * (1 - t_local) + valences[track_idx + 1] * t_local
                arousal = arousals[track_idx] * (1 - t_local) + arousals[track_idx + 1] * t_local
                color1 = np.array(colors[track_idx])
                color2 = np.array(colors[min(track_idx + 1, n_tracks - 1)])
                raw_color = color1 * (1 - t_local) + color2 * t_local
            else:
                valence = valences[track_idx]
                arousal = arousals[track_idx]
                raw_color = np.array(colors[track_idx])

            smoothed_color = smoothed_color * (1 - color_smoothing_factor) + raw_color * color_smoothing_factor
            current_color = smoothed_color.astype(int)

            frame = np.zeros((height, width, 3), dtype=np.uint8)
            bg_brightness = 0.25
            for c in range(3):
                frame[:, :, c] = (current_color[c] * bg_intensity * (bg_brightness / 0.15)).astype(np.uint8)

            target_x = center_x + (valence - 0.5) * max_radius * 1.4
            target_y = center_y - (arousal - 0.5) * max_radius * 1.4

            direction_x = target_x - smoothed_focal_x
            direction_y = target_y - smoothed_focal_y
            distance = np.sqrt(direction_x**2 + direction_y**2)
            
            if distance > 0:
                direction_x /= distance
                direction_y /= distance
                move_distance = min(focal_speed, distance)
                smoothed_focal_x += direction_x * move_distance
                smoothed_focal_y += direction_y * move_distance

            x_focal = int(smoothed_focal_x)
            y_focal = int(smoothed_focal_y)

            if frame_idx % 6 == 0:
                voronoi_points += voronoi_velocities

                directions = np.array([x_focal, y_focal]) - voronoi_points
                dists = np.linalg.norm(directions, axis=1, keepdims=True)
                dists[dists == 0] = 1
                voronoi_velocities += directions / dists * 0.01

                voronoi_velocities[:, 0] *= np.where(
                    (voronoi_points[:, 0] < 0) | (voronoi_points[:, 0] >= width), -1, 1
                )
                voronoi_velocities[:, 1] *= np.where(
                    (voronoi_points[:, 1] < 0) | (voronoi_points[:, 1] >= height), -1, 1
                )
                voronoi_points = np.clip(voronoi_points, [0, 0], [width-1, height-1])

                variations = (np.sin(np.arange(n_voronoi) * 0.5 + frame_idx * 0.01) + 1) * 0.5
                voronoi_colors = [
                    (current_color * (0.4 + v * 0.3)).astype(int)
                    for v in variations
                ]
                
                voronoi_cache = self._create_voronoi_pattern_fast(
                    width, height, voronoi_points, voronoi_colors
                )
                voronoi_cache_counter = 0

            if voronoi_cache is not None:
                frame = cv2.addWeighted(frame, 1.0, voronoi_cache, 0.35, 0)
                voronoi_cache_counter += 1

            rotation_speed = arousal * 0.03 + 0.01
            base_rotation = frame_idx * rotation_speed

            if valence < 0.33:
                sides = 3
            elif valence < 0.66:
                sides = 4
            else:
                sides = 6
            sides += int(arousal * 3)

            for layer in range(3):
                layer_scale = 1 - layer * 0.2
                layer_alpha = (1 - layer * 0.2) * 0.85

                radius = max_radius * layer_scale * (0.3 + valence * 0.4)
                rotation = base_rotation * (1 + layer * 0.3) + layer * np.pi / sides

                angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + rotation
                points = np.array([
                    [x_focal + radius * np.cos(a), y_focal + radius * np.sin(a)]
                    for a in angles
                ], dtype=np.int32)
                
                poly_color = tuple((current_color * layer_alpha).astype(int).tolist())
                cv2.polylines(frame, [points], True, poly_color, 2, cv2.LINE_AA)

                if layer == 1:
                    inner_radius = radius * 0.7
                    inner_rotation = -rotation * 1.5
                    inner_points = np.array([
                        [x_focal + inner_radius * np.cos(a), y_focal + inner_radius * np.sin(a)]
                        for a in np.linspace(0, 2 * np.pi, sides, endpoint=False) + inner_rotation
                    ], dtype=np.int32)
                    inner_color = tuple((current_color * layer_alpha * 0.75).astype(int).tolist())
                    cv2.polylines(frame, [inner_points], True, inner_color, 1, cv2.LINE_AA)

            circle_phases = (frame_idx * 0.005 + np.arange(5) * 0.4) % (2 * np.pi)
            circle_alphas = (np.sin(circle_phases) + 1) * 0.5 * 0.55 + 0.05
            for i, alpha in enumerate(circle_alphas):
                circle_radius = int(max_radius * (0.25 + i * 0.12) * (0.8 + arousal * 0.4))
                circle_color = tuple((current_color * alpha).astype(int).tolist())
                cv2.circle(frame, (x_focal, y_focal), circle_radius, circle_color, 1, cv2.LINE_AA)

            orbit_speed = arousal * 0.04 + 0.02
            orbit_angles += orbit_speed * (1 + np.sin(np.arange(n_particles_orbit) * 0.1) * 0.5)

            orbit_layers = (np.arange(n_particles_orbit) / (n_particles_orbit / 3)).astype(int)
            orbit_bases = max_radius * (0.3 + orbit_layers * 0.2)
            
            particles_orbit_x = x_focal + orbit_radii * orbit_bases / max_radius * np.cos(orbit_angles)
            particles_orbit_y = y_focal + orbit_radii * orbit_bases / max_radius * np.sin(orbit_angles)
            particles_orbit_pos = np.column_stack((particles_orbit_x, particles_orbit_y)).astype(int)

            particle_size = 2 + int(arousal * 2)
            orbit_color = tuple((current_color * 1.0).astype(int).tolist())
            for x, y in particles_orbit_pos:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), particle_size, orbit_color, -1, cv2.LINE_AA)

            focal_point = np.array([x_focal, y_focal])
            focal_radius = 40 + arousal * 30
            constant_speed = 4.0

            particles_flow += velocities_flow

            to_focal = particles_flow - focal_point
            distances_to_focal = np.linalg.norm(to_focal, axis=1)
            collision_mask = distances_to_focal < focal_radius
            
            if np.any(collision_mask):
                collision_directions = to_focal[collision_mask]
                collision_distances = distances_to_focal[collision_mask, None]
                collision_distances[collision_distances == 0] = 1

                normal_vectors = collision_directions / collision_distances

                dot_products = np.sum(velocities_flow[collision_mask] * normal_vectors, axis=1, keepdims=True)
                reflected_velocities = velocities_flow[collision_mask] - 2 * dot_products * normal_vectors

                reflected_speeds = np.linalg.norm(reflected_velocities, axis=1, keepdims=True)
                reflected_speeds[reflected_speeds == 0] = 1
                velocities_flow[collision_mask] = (reflected_velocities / reflected_speeds) * constant_speed

                push_distance = focal_radius - distances_to_focal[collision_mask, None] + 5
                particles_flow[collision_mask] += normal_vectors * push_distance

            boundary_mask_x_min = particles_flow[:, 0] < 0
            boundary_mask_x_max = particles_flow[:, 0] >= width
            boundary_mask_y_min = particles_flow[:, 1] < 0
            boundary_mask_y_max = particles_flow[:, 1] >= height
            
            velocities_flow[boundary_mask_x_min | boundary_mask_x_max, 0] *= -1
            velocities_flow[boundary_mask_y_min | boundary_mask_y_max, 1] *= -1

            particles_flow[:, 0] = np.clip(particles_flow[:, 0], 0, width - 1)
            particles_flow[:, 1] = np.clip(particles_flow[:, 1], 0, height - 1)

            current_speeds = np.linalg.norm(velocities_flow, axis=1, keepdims=True)
            current_speeds[current_speeds == 0] = 1
            velocities_flow = (velocities_flow / current_speeds) * constant_speed

            particle_history.append(particles_flow.copy())
            if len(particle_history) > 10:
                particle_history.pop(0)

            for trail_idx, trail_particles in enumerate(particle_history[::2]):
                alpha = (trail_idx * 2 + 1) / len(particle_history)
                trail_color = tuple((current_color * alpha * 0.35).astype(int).tolist())
                trail_size = 1 + int(alpha * 2)
                for particle in trail_particles[::4]:
                    x, y = int(particle[0]), int(particle[1])
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), trail_size, trail_color, -1)

            flow_size = 2 + int(valence * 2)
            flow_color = tuple((current_color * 1.1).astype(int).tolist())
            for particle in particles_flow[::2]:
                x, y = int(particle[0]), int(particle[1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), flow_size, flow_color, -1, cv2.LINE_AA)

            radial_angles += arousal * 0.01
            radial_phases = (radial_phases + radial_speeds * arousal * 0.5) % max_radius

            radial_x = x_focal + radial_phases * np.cos(radial_angles)
            radial_y = y_focal + radial_phases * np.sin(radial_angles)
            radial_positions = np.column_stack((radial_x, radial_y)).astype(int)
            
            intensities = 1 - (radial_phases / max_radius)

            for i, (x, y) in enumerate(radial_positions[::2]):
                if 0 <= x < width and 0 <= y < height:
                    intensity = intensities[i * 2]
                    size = 1 + int(intensity * 3)
                    particle_color = tuple((current_color * intensity * 1.0).astype(int).tolist())
                    cv2.circle(frame, (x, y), size, particle_color, -1)

            focal_size = int(60 + arousal * 40)
            pulse = (np.sin(frame_idx * 0.03) + 1) * 0.5 * 0.05 + 0.95

            y1, y2 = max(0, y_focal - focal_size), min(height, y_focal + focal_size + 1)
            x1, x2 = max(0, x_focal - focal_size), min(width, x_focal + focal_size + 1)
            
            if y2 > y1 and x2 > x1:
                yy, xx = np.ogrid[y1-y_focal:y2-y_focal, x1-x_focal:x2-x_focal]
                dist_grid = np.sqrt(xx*xx + yy*yy)

                alpha_grid = np.clip(1 - (dist_grid / focal_size), 0, 1) ** 1.2
                alpha_grid *= pulse * 1.3

                for c in range(3):
                    glow = (current_color[c] * alpha_grid).astype(np.uint8)
                    frame[y1:y2, x1:x2, c] = np.clip(
                        frame[y1:y2, x1:x2, c].astype(int) + glow, 0, 255
                    ).astype(np.uint8)

            if frame_idx % 3 == 0:
                connection_threshold = 80 + arousal * 100
                line_color = tuple((current_color * 0.25).astype(int).tolist())

                sample_particles = particles_flow[::5][:20]
                for i in range(len(sample_particles)):
                    for j in range(i + 1, len(sample_particles)):
                        dist = np.linalg.norm(sample_particles[i] - sample_particles[j])
                        if dist < connection_threshold:
                            p1 = tuple(sample_particles[i].astype(int))
                            p2 = tuple(sample_particles[j].astype(int))
                            cv2.line(frame, p1, p2, line_color, 1, cv2.LINE_AA)
            
            frames.append(frame)

        output_path = save_path if save_path else self.output_dir / "mood_video.mp4"
        
        logger.info("Encoding video to %s", output_path)
        imageio.mimwrite(
            str(output_path),
            frames,
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        
        logger.info("Mood video saved to %s", output_path)
        return output_path
