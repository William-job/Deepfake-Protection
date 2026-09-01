import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.models.car import CAR


def visualize_artifact_composition(model, val_loader, device, output_path):
    model.eval()
    compositions = {"temporal": [], "flow": [], "frequency": [], "blending": []}
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting compositions"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            outputs = model(frames)
            w = outputs["w"]

            for i, key in enumerate(["temporal", "flow", "frequency", "blending"]):
                compositions[key].extend(w[:, i].cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    artifact_names = {
        "temporal": "Temporal Inconsistency",
        "flow": "Flow Linearity Deviation",
        "frequency": "Frequency Aliasing",
        "blending": "Boundary Blending",
    }

    labels_arr = np.array(labels_list)
    for i, (key, name) in enumerate(artifact_names.items()):
        vals = np.array(compositions[key])
        real_vals = vals[labels_arr == 0]
        fake_vals = vals[labels_arr == 1]

        axes[i].hist(real_vals, bins=30, alpha=0.6, label="Real", color="green", density=True)
        axes[i].hist(fake_vals, bins=30, alpha=0.6, label="Fake", color="red", density=True)
        axes[i].set_title(name)
        axes[i].set_xlabel("Weight")
        axes[i].set_ylabel("Density")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Artifact composition visualization saved to {output_path}")


def visualize_latent_space(model, val_loader, device, output_path, max_samples=500):
    model.eval()
    latent_vectors = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting latent vectors"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            outputs = model(frames)
            z = outputs["z"]

            latent_vectors.append(z.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            if len(np.concatenate(latent_vectors)) >= max_samples:
                break

    z_all = np.concatenate(latent_vectors)[:max_samples]
    labels_all = np.concatenate(labels_list)[:max_samples]

    if z_all.shape[0] < 10:
        print("Not enough samples for t-SNE visualization")
        return

    perplexity = min(30, z_all.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    z_2d = tsne.fit_transform(z_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all, cmap="coolwarm",
                          alpha=0.7, s=20)
    plt.colorbar(scatter, label="Label (0=Real, 1=Fake)")
    plt.title("t-SNE Visualization of Artifact Latent Space")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE visualization saved to {output_path}")


def visualize_confusion_matrix(model, val_loader, device, output_path):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            outputs = model(frames)
            logits = outputs["logits"]
            if logits.size(1) > 1:
                preds = torch.sigmoid(logits[:, 1])
            else:
                preds = torch.sigmoid(logits.squeeze(-1))

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    preds_binary = (np.array(all_preds) >= 0.5).astype(int)
    labels_binary = np.array(all_labels).astype(int)

    cm = confusion_matrix(labels_binary, preds_binary)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - CAR Deepfake Detection")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CAR Model Visualization")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.checkpoint}...")
    model = CAR(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loading validation data...")
    val_loader = create_dataloader(config, split="val")

    visualize_artifact_composition(
        model, val_loader, device,
        os.path.join(args.output_dir, "artifact_composition.png")
    )

    visualize_latent_space(
        model, val_loader, device,
        os.path.join(args.output_dir, "latent_tsne.png")
    )

    visualize_confusion_matrix(
        model, val_loader, device,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()