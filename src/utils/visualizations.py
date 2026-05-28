import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from torchvision.utils import make_grid


def build_reconstruction_grid(images, reconstructions, max_images=16):
    images = images[:max_images]
    reconstructions = reconstructions[:max_images]

    if images.size(0) == 0:
        return None

    grid = make_grid(
        torch.cat([images, reconstructions], dim=0),
        nrow=images.size(0),
        padding=2,
    )

    return grid


def build_test_grid(images, reconstructions, labels, max_images=16):
    if images.size(0) == 0:
        return None

    good_idx = (labels == 0).nonzero(as_tuple=True)[0]
    defect_idx = (labels == 1).nonzero(as_tuple=True)[0]

    if good_idx.numel() == 0 and defect_idx.numel() == 0:
        return None

    num_each = max_images // 2
    #good_indices = torch.randperm(good_idx.size(0))
    #defect_indices = torch.randperm(defect_idx.size(0))
    #good_idx_shuffled = good_idx[good_indices]
    #defect_idx_shuffled = defect_idx[defect_indices]
    selected = torch.cat(
        [good_idx[:num_each], defect_idx[:num_each]],
        dim=0,
    )

    if selected.numel() == 0:
        return None

    images_sel = images[selected]
    recon_sel = reconstructions[selected]

    grid = make_grid(
        torch.cat([images_sel, recon_sel], dim=0),
        nrow=images_sel.size(0),
        padding=2,
    )

    return grid


def plot_tsne(z, labels, max_points=500, random_state=42):
    if z.size(0) < 2:
        return None

    if z.size(0) < 5:
        return None

    if z.size(0) > max_points:
        idx = torch.randperm(z.size(0))[:max_points]
        z = z[idx]
        labels = [labels[i] for i in idx.tolist()]

    z_np = z.cpu().numpy()
    labels_np = np.array(labels)

    perplexity = min(5, z_np.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_state,
    )

    embedding = tsne.fit_transform(z_np)

    fig, ax = plt.subplots(figsize=(6, 4))

    for label in sorted(set(labels_np)):
        mask = labels_np == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.8,
            label=str(label),
        )

    ax.set_title("t-SNE latent space (val)")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    return fig

def plot_tsne_wandb(z, labels, max_points=500, random_state=42):
    if z.size(0) < 2:
        return None

    if z.size(0) < 5:
        return None

    if z.size(0) > max_points:
        idx = torch.randperm(z.size(0))[:max_points]
        z = z[idx]
        labels = [labels[i] for i in idx.tolist()]

    z_np = z.cpu().numpy()
    labels_np = np.array(labels)

    perplexity = min(5, z_np.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_state,
    )

    embedding = tsne.fit_transform(z_np)

    columns = ["x", "y"]
    if labels is not None:
        columns.append("label")

    table = wandb.Table(columns=columns)

    for i, (x, y) in enumerate(embedding):
        row = [x, y]
        if labels is not None:
            label = labels[i]
            row.append(label)
        table.add_data(*row)

    for label in sorted(set(labels_np)):
        mask = labels_np == label

    return table


def plot_error_histograms(good_errors, defect_errors_by_type, bins=30):
    figs = []

    if not good_errors:
        return figs

    for defect_type, errors in defect_errors_by_type.items():
        if not errors:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(good_errors, bins=bins, alpha=0.6, label="good")
        ax.hist(errors, bins=bins, alpha=0.6, label=defect_type)
        ax.set_title(f"Reconstruction error: good vs {defect_type}")
        ax.set_xlabel("reconstruction error")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()

        figs.append((defect_type, fig))

    return figs

def plot_error_histograms_wandb(histogram_groups, bins=30):
    figs = []

    if len(histogram_groups) < 1:
        return figs

    for class_name, errors_dict in histogram_groups.items():

        for defect_type, errors in errors_dict["defect_errors"].items():
            data = [errors_dict["good_errors"], errors]
            table = wandb.Table(data=data, columns=["Good error", "Defect error"])
            figs.append((class_name, defect_type, table))

    return figs
