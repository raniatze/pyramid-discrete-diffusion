import torch
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def load_checkpoint_metrics(checkpoint_path):
    """Load training metrics from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        train_metrics = checkpoint.get('train_metrics', {})

        if not train_metrics:
            print(f"Warning: No training metrics found in {checkpoint_path}")
            return None

        return train_metrics
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def plot_training_metrics(metrics_dict, save_path=None, title="Training Progress"):
    """Plot training metrics with a clean, modern style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different metrics
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']

    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        epochs = range(1, len(values) + 1)
        color = colors[i % len(colors)]

        ax.plot(epochs, values,
                label=metric_name.replace('_', ' ').title(),
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.8)

    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Make it look clean
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def find_latest_checkpoint(models_dir):
    """Find the latest checkpoint in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None

    checkpoints = list(models_path.glob("epoch*.tar"))
    if not checkpoints:
        return None

    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.replace('epoch', '')))
    return str(checkpoints[-1])


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="/home/raniatze/Documents/PhD/Research/pyramid-discrete-diffusion/checkpoints/s_2_to_s_3_cnt/epoch220.tar")
    parser.add_argument("--models_dir", type=str, help="Directory containing checkpoint files")
    parser.add_argument("--save", type=str, help="Path to save the plot")
    parser.add_argument("--title", type=str, default="Training Loss Progress", help="Plot title")

    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.models_dir:
        checkpoint_path = find_latest_checkpoint(args.models_dir)
        if checkpoint_path:
            print(f"Using latest checkpoint: {checkpoint_path}")

    if not checkpoint_path:
        print("Error: Please provide either --checkpoint or --models_dir")
        return

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    # Load and plot metrics
    metrics = load_checkpoint_metrics(checkpoint_path)
    if metrics:
        print(f"Loaded metrics: {list(metrics.keys())}")
        plot_training_metrics(metrics, save_path=args.save, title=args.title)
    else:
        print("No metrics to plot.")


# Quick usage function for interactive use
def quick_plot(checkpoint_path, save_path=None):
    """Quick function to plot metrics from a checkpoint."""
    metrics = load_checkpoint_metrics(checkpoint_path)
    if metrics:
        plot_training_metrics(metrics, save_path=save_path)
        return metrics
    return None


if __name__ == "__main__":
    main()