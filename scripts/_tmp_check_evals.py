import json

for m in ["xception", "mesonet", "efficientnet_b0"]:
    for s in [42, 43, 44]:
        p = f"results/baseline_honest/{m}/seed_{s}/eval_metrics.json"
        try:
            d = json.load(open(p))
            print(f"{m}/seed_{s}: auc={d['auc']:.4f} "
                  f"val={d['val_threshold_info']['val_auc']:.4f} "
                  f"ts={d['timestamp']} ckpt={d['checkpoint'][:45]}")
        except Exception as e:
            print(f"{m}/seed_{s}: MISSING ({e.__class__.__name__})")
