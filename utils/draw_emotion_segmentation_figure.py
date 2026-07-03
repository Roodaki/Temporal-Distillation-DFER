import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy
from tqdm import tqdm

# ======================================
# CONFIGURATION
# ======================================
root_dir = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Video Facial Emotion Recognition (VFER)\Dataset\rotated_face224_analyze"  # 🔸 update this path

# Configuration for automatic peak segment detection
MIN_CONSECUTIVE_FRAMES = 1
CLASS_TO_EMOTION_MAP = {
    "amusement": "happy",
    "anger": "angry",
    "awe": "surprise",
    "disgust": "disgust",
    "enthusiasm": "happy",
    "fear": "fear",
    "liking": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

# Emotion color palette (soft yet distinct)
emotion_colors = {
    "angry": "#E24A33",
    "disgust": "#9467BD",
    "fear": "#8C564B",
    "happy": "#FFD700",
    "sad": "#1F77B4",
    "surprise": "#2CA02C",
    "neutral": "#7F7F7F",
}

# Global plotting settings
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "xtick.color": "#444",
        "ytick.color": "#444",
    }
)


# ======================================
# FUNCTION: Compute advanced diversity score - UNCHANGED
# ======================================
def compute_diversity(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.shape[0] < 2:
            return -1, None
        if "frame" not in df.columns:
            df.insert(0, "frame", range(len(df)))
        df["dominant"] = df.iloc[:, 1:].idxmax(axis=1)
        emotion_counts = df["dominant"].value_counts(normalize=True)
        shannon_entropy = entropy(emotion_counts, base=2)
        transitions = max((df["dominant"] != df["dominant"].shift()).sum() - 1, 0)
        transition_rate = transitions / (len(df) - 1)
        diversity_score = shannon_entropy * (1 + transition_rate)
        metrics = {
            "score": diversity_score,
            "entropy": shannon_entropy,
            "transition_rate": transition_rate,
        }
        return metrics, df
    except Exception as e:
        print(f"⚠️ Skipping {csv_path} due to error: {e}")
        return -1, None


# ======================================
# HELPER FUNCTIONS - UNCHANGED
# ======================================
def get_target_emotion_from_path(csv_path):
    try:
        folder_name = os.path.basename(os.path.dirname(csv_path))
        emotion_class = folder_name.split("_")[1]
        return CLASS_TO_EMOTION_MAP.get(emotion_class)
    except (IndexError, KeyError):
        return None


def find_peak_emotion_segments(df, target_emotion, min_length):
    if not target_emotion or df.empty:
        return []
    peak_segments = []
    is_target = df["dominant"] == target_emotion
    change_points = is_target.ne(is_target.shift()).cumsum()
    target_blocks = df[is_target].groupby(change_points)
    for _, block_df in target_blocks:
        if len(block_df) >= min_length:
            peak_segments.append((block_df["frame"].min(), block_df["frame"].max()))
    return peak_segments


# ======================================
# ⬇️ MODIFIED PLOTTING FUNCTION
# ======================================
def plot_timeline_with_masked_segments(
    df, peak_segments, target_emotion, save_path=None
):
    """
    Creates a two-part plot:
    1. Top: A frame-by-frame "barcode" of emotions.
    2. Bottom: Aggregated segments with non-peak segments masked.
    """
    bg_color, text_color, grid_color = "#F8F9FA", "#222", "#DDD"
    fig, (ax_full, ax_masked) = plt.subplots(
        2, 1, figsize=(12, 4.2), facecolor=bg_color, sharex=True
    )
    fig.subplots_adjust(hspace=0.4, right=0.85)

    # --- 1. Top Plot (Frame-by-frame barcode) ---
    ax_full.set_facecolor(bg_color)
    # Get colors for each frame
    colors = df["dominant"].map(emotion_colors).fillna("gray")
    # Plot all frames as vertical lines
    ax_full.vlines(df["frame"], ymin=0, ymax=1, colors=colors, linewidth=1)
    ax_full.set_title("Frame-by-Frame Emotion Distribution", fontsize=11)

    # --- 2. Bottom Plot (Masked aggregated segments) ---
    ax_masked.set_facecolor(bg_color)

    # Helper function to draw aggregated bars for the bottom plot
    def _draw_aggregated_bars(ax, alpha_masking=False):
        segments = []
        current_emotion, start_frame = df["dominant"].iloc[0], df["frame"].iloc[0]
        for i in range(1, len(df)):
            if df["dominant"].iloc[i] != current_emotion:
                segments.append((start_frame, df["frame"].iloc[i - 1], current_emotion))
                start_frame, current_emotion = (
                    df["frame"].iloc[i],
                    df["dominant"].iloc[i],
                )
        segments.append((start_frame, df["frame"].iloc[-1], current_emotion))

        for start, end, emotion in segments:
            alpha = 1.0
            if alpha_masking:
                is_peak = any(
                    start <= p_end and end >= p_start
                    for p_start, p_end in peak_segments
                )
                if not is_peak:
                    alpha = 0.1
            ax.broken_barh(
                [(start, end - start)],
                (0, 1),
                facecolors=emotion_colors.get(emotion, "gray"),
                lw=0,
                alpha=alpha,
            )

    _draw_aggregated_bars(ax_masked, alpha_masking=True)
    ax_masked.set_title(
        f"Timeline with Peak Segments (Min. {MIN_CONSECUTIVE_FRAMES} Frames) Highlighted",
        fontsize=11,
    )

    # Common Formatting for both axes
    for ax in [ax_full, ax_masked]:
        ax.set_xlim(df["frame"].min(), df["frame"].max())
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=text_color, bottom=True, labelbottom=True)
        ax.grid(axis="x", color=grid_color, linestyle="--", linewidth=0.5, alpha=0.4)
    ax_masked.set_xlabel("Frame", fontsize=10, color=text_color, labelpad=6)
    # Hide x-axis labels on the top plot to avoid overlap
    plt.setp(ax_full.get_xticklabels(), visible=False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=e.capitalize()) for e, c in emotion_colors.items()
    ]
    ax_full.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.05),
        loc="upper left",
        frameon=False,
        title="Emotion",
        title_fontsize=10,
        labelcolor=text_color,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=bg_color)
        print(f"💾 Saved plot to: {save_path}")
    plt.show()


# ======================================
# ORIGINAL FUNCTION: Plot dominant emotion timeline - UNCHANGED
# ======================================
def plot_emotion_timeline(df, save_path=None, dark_mode=False):
    pass  # Original code would be here


# ======================================
# MAIN LOOP: Find the most diverse video - UNCHANGED
# ======================================
most_diverse_file = None
highest_score = -1
most_diverse_df = None
most_diverse_metrics = None
csv_files = [
    os.path.join(r, f)
    for r, _, fs in os.walk(root_dir)
    for f in fs
    if f.endswith(".csv")
]
if not csv_files:
    print("⚠️ No valid CSV files found.")
    exit()
for csv_path in tqdm(csv_files, desc="Processing CSVs"):
    metrics, df = compute_diversity(csv_path)
    if isinstance(metrics, dict) and metrics["score"] > highest_score:
        highest_score = metrics["score"]
        most_diverse_file = csv_path
        most_diverse_df = df
        most_diverse_metrics = metrics

# ======================================
# MODIFIED OUTPUT RESULT - UNCHANGED
# ======================================
if most_diverse_file:
    print(f"\n🎬 Most diverse video found:\n   {most_diverse_file}")
    print(
        f"📊 Shannon Entropy: {most_diverse_metrics['entropy']:.4f}\n"
        f"🔄 Transition Rate: {most_diverse_metrics['transition_rate']:.4f}\n"
        f"📈 Diversity Score: {most_diverse_metrics['score']:.4f}\n"
    )

    target_emotion = get_target_emotion_from_path(most_diverse_file)
    print(f"🎯 Target emotion from file path: '{target_emotion}'")

    if target_emotion:
        peak_segments = find_peak_emotion_segments(
            most_diverse_df, target_emotion, MIN_CONSECUTIVE_FRAMES
        )
        if peak_segments:
            print(
                f"🔍 Found {len(peak_segments)} peak segments (>= {MIN_CONSECUTIVE_FRAMES} frames)."
            )
            save_path = (
                os.path.splitext(most_diverse_file)[0] + "_peak_segments_timeline.png"
            )
            plot_timeline_with_masked_segments(
                most_diverse_df, peak_segments, target_emotion, save_path=save_path
            )
        else:
            print(
                f"⚠️ No segments of '{target_emotion}' found meeting the {MIN_CONSECUTIVE_FRAMES}-frame minimum."
            )
    else:
        print("⚠️ Could not determine target emotion from file path.")
else:
    print("⚠️ No valid CSV files were successfully processed.")
