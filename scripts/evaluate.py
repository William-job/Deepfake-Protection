import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.config import load_config
from src.data.dataloader import create_dataloader
from src.models.car import CAR
from src.utils.metrics import compute_all_metrics


def ablation_study(model, val_loader, device="cuda"):
    model.eval()

    expert_names = ["temporal", "flow", "frequency", "blending"]
    ablation_results = {}

    all_preds_full = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Full model evaluation"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            outputs = model(frames)
            logits = outputs["logits"]
            if logits.size(1) > 1:
                preds = torch.sigmoid(logits[:, 1])
            else:
                preds = torch.sigmoid(logits.squeeze(-1))

            all_preds_full.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    full_metrics = compute_all_metrics(np.array(all_preds_full), np.array(all_labels))
    ablation_results["full_model"] = full_metrics

    for skip_expert in expert_names:
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ablation: skip {skip_expert}"):
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device)

                outputs = model(frames)
                w = outputs["w"]

                expert_logits = []
                for i, name in enumerate(expert_names):
                    if name == skip_expert:
                        continue
                    expert_out = model.experts[name](outputs["head_outputs"][name])
                    expert_logits.append(expert_out)

                if expert_logits:
                    w_filtered = torch.zeros_like(w)
                    kept_idx = 0
                    for i, name in enumerate(expert_names):
                        if name != skip_expert:
                            w_filtered[:, i] = w[:, i]
                            kept_idx += 1
                    w_filtered = w_filtered / (w_filtered.sum(dim=-1, keepdim=True) + 1e-8)

                    expert_logits = torch.stack(expert_logits, dim=1)
                    y_combined = (w_filtered.unsqueeze(-1) * expert_logits).sum(dim=1)
                else:
                    y_combined = torch.zeros(labels.size(0), 2, device=device)

                if y_combined.size(1) > 1:
                    preds = torch.sigmoid(y_combined[:, 1])
                else:
                    preds = torch.sigmoid(y_combined.squeeze(-1))

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        ablation_results[f"skip_{skip_expert}"] = compute_all_metrics(
            np.array(all_preds), np.array(all_labels)
        )

    return ablation_results


def print_ablation_results(results):
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Configuration':<25} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'AP':>8}")
    print("-" * 70)

    for config, metrics in results.items():
        print(f"{config:<25} {metrics['auc']:>8.4f} {metrics['accuracy']:>10.4f} "
              f"{metrics['f1']:>8.4f} {metrics['ap']:>8.4f}")

    print("-" * 70)

    full_auc = results["full_model"]["auc"]
    for config, metrics in results.items():
        if config != "full_model":
            drop = full_auc - metrics["auc"]
            print(f"  AUC drop without {config.replace('skip_', '')}: {drop:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CAR Model Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

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

    if args.ablation:
        results = ablation_study(model, val_loader, device)
        print_ablation_results(results)
    else:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
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

        metrics = compute_all_metrics(np.array(all_preds), np.array(all_labels))

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AP:        {metrics['ap']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()