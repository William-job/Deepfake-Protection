import json
import time
import torch
import cv2
import numpy as np
from collections import deque


class CARInferencePipeline:
    def __init__(self, model, config, device="cuda", use_amp=False):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.use_amp = use_amp and device != "cpu"

        self.num_frames = config.data.num_frames
        self.frame_stride = config.data.frame_stride
        self.image_size = config.data.image_size

        self.frame_buffer = deque(maxlen=self.num_frames * self.frame_stride)

    def _preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        return frame

    def _build_batch(self):
        frames = list(self.frame_buffer)
        indices = list(range(0, len(frames), self.frame_stride))[:self.num_frames]
        sampled = [frames[i] for i in indices]

        while len(sampled) < self.num_frames:
            sampled.append(sampled[-1] if sampled else torch.zeros(3, self.image_size, self.image_size))

        batch = torch.stack(sampled, dim=0)
        batch = batch.unsqueeze(0)
        return batch

    @torch.no_grad()
    def predict_frame(self, frame):
        start_time = time.time()

        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)

        batch = self._build_batch().to(self.device)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                pred, report = self.model.predict(batch)
        else:
            pred, report = self.model.predict(batch)

        latency = (time.time() - start_time) * 1000

        report["latency_ms"] = round(latency, 2)

        return pred.item(), report

    @torch.no_grad()
    def predict_video(self, video_path, output_path=None, real_time=False):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []
        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pred, report = self.predict_frame(frame)
            report["frame"] = frame_idx
            results.append(report)

            frame_idx += 1

            if real_time:
                label = "FAKE" if pred >= 0.5 else "REAL"
                color = (0, 0, 255) if pred >= 0.5 else (0, 255, 0)
                cv2.putText(frame, f"{label}: {pred:.3f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("CAR Deepfake Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if real_time:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        actual_fps = frame_idx / elapsed if elapsed > 0 else 0

        summary = {
            "video_path": video_path,
            "total_frames": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(actual_fps, 2),
            "fake_ratio": round(sum(1 for r in results if r["prediction"] == "fake") / max(len(results), 1), 4),
            "results": results,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    @torch.no_grad()
    def predict_batch(self, frames_batch):
        if isinstance(frames_batch, torch.Tensor):
            batch = frames_batch.to(self.device)
        else:
            batch = torch.stack([
                self._preprocess_frame(f) for f in frames_batch
            ]).unsqueeze(0).to(self.device)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                pred, report = self.model.predict(batch)
        else:
            pred, report = self.model.predict(batch)

        return pred.item(), report
