"""Neural Refiner — small PyTorch network that refines EmotionNet logits.

Wraps around the existing rule-based feature extraction and adds a learned
nonlinear transformation. Falls back to the original logits when PyTorch
is unavailable or no trained weights exist.

Architecture:
  Input:  27-dim feature vector from EmotionNet._extract_features()
  Hidden: 64 -> ReLU -> Dropout(0.1) -> 32 -> ReLU
  Output: 11 logits (one per emotion)

Training:
  Self-supervised from conversation history — the refiner learns to predict
  the final blended emotion distribution from the raw features. Training
  happens incrementally on every inference call (online learning) so the
  network adapts to the user's conversation style over time.

Persistence:
  Weights saved to data/neural_refiner.pt (if PyTorch available).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_NEURAL_REFINER_DIR = Path(__file__).parent / "data"
_NEURAL_REFINER_PATH = _NEURAL_REFINER_DIR / "neural_refiner.pt"
_NEURAL_REFINER_META = _NEURAL_REFINER_DIR / "neural_refiner_meta.json"

# Feature vector dimension (must match EmotionNet._extract_features keys)
_FEATURE_KEYS = [
    "sentiment", "positive", "negative", "gratitude", "frustration",
    "anger", "sadness", "anxiety", "loneliness", "curiosity", "certainty",
    "urgency", "question", "exclaim", "caps", "length", "signal_strength",
    "topic_sensitivity", "topic_importance", "profile_warmth",
    "profile_assertiveness", "long_pause", "has_greeting", "has_name",
    "is_short", "has_question_mark", "has_ellipsis",
]
_FEATURE_DIM = len(_FEATURE_KEYS)
_NUM_EMOTIONS = 11

_torch_available = False
_nn = None
_torch = None

try:
    import torch as _torch_mod
    import torch.nn as _nn_mod
    _torch = _torch_mod
    _nn = _nn_mod
    _torch_available = True
except ImportError:
    pass


if _torch_available:
    class EmotionRefinerNet(_nn.Module):
        """Small feedforward network that refines emotion logits."""

        def __init__(self, feature_dim: int = _FEATURE_DIM, num_emotions: int = _NUM_EMOTIONS):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(feature_dim, 64),
                _nn.ReLU(),
                _nn.Dropout(0.1),
                _nn.Linear(64, 32),
                _nn.ReLU(),
                _nn.Linear(32, num_emotions),
            )
            self._init_weights()

        def _init_weights(self):
            # Initialize near-identity so the network starts by passing
            # through the rule-based logits with minimal modification.
            for m in self.net:
                if isinstance(m, _nn.Linear):
                    _nn.init.xavier_uniform_(m.weight, gain=0.1)
                    _nn.init.zeros_(m.bias)

        def forward(self, features: _torch.Tensor) -> _torch.Tensor:
            return self.net(features)


class NeuralRefiner:
    """Manages the PyTorch neural refiner for EmotionNet.

    Falls back gracefully when PyTorch is not available.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg: None)
        self._lock = threading.Lock()
        self._model = None
        self._optimizer = None
        self._step_count = 0
        self._online_lr = 0.001
        self._last_inference_ms = 0.0
        self._refinement_enabled = True
        self._training_enabled = True
        self._cumulative_loss = 0.0

        if _torch_available:
            try:
                self._model = EmotionRefinerNet()
                self._optimizer = _torch.optim.Adam(
                    self._model.parameters(), lr=self._online_lr
                )
                self._load_weights()
                self._log("[NeuralRefiner] PyTorch refiner initialized (online learning active)")
            except Exception as exc:
                self._log(f"[NeuralRefiner] PyTorch init failed: {exc} — using rule-based fallback")
                self._model = None
        else:
            self._log("[NeuralRefiner] PyTorch not available — using rule-based EmotionNet only")

    @property
    def available(self) -> bool:
        return self._model is not None and _torch_available

    @property
    def is_training(self) -> bool:
        return self._training_enabled and self.available

    def refine_logits(self, features: dict[str, float], base_logits: dict[str, float]) -> dict[str, float]:
        """Refine the rule-based logits using the neural network.

        Args:
            features: Feature vector from EmotionNet._extract_features()
            base_logits: Rule-based logits from EmotionNet

        Returns:
            Refined logits dict, or the original base_logits if unavailable.
        """
        if not self.available or not self._refinement_enabled:
            return dict(base_logits)

        t0 = time.perf_counter()
        try:
            # Build feature tensor in canonical key order
            feat_vec = [float(features.get(k, 0.0)) for k in _FEATURE_KEYS]
            feat_tensor = _torch.tensor([feat_vec], dtype=_torch.float32)

            with _torch.no_grad():
                residual = self._model(feat_tensor).squeeze(0)

            # Apply residual: refined = base + small residual from network
            # Scale residual by 0.3 so the network can only *refine*, not *replace*
            emotions = list(base_logits.keys())
            refined = {}
            for i, emo in enumerate(emotions):
                base_val = float(base_logits[emo])
                res_val = float(residual[i]) * 0.3  # Scale down residual
                refined[emo] = base_val + res_val

            ms = (time.perf_counter() - t0) * 1000.0
            self._last_inference_ms = ms
            return refined

        except Exception as exc:
            self._log(f"[NeuralRefiner] Refinement failed: {exc}")
            return dict(base_logits)

    def online_learn(self, features: dict[str, float], target_probs: dict[str, float]):
        """Perform one step of online learning.

        Uses the final blended emotion distribution as the target, so the
        network gradually learns to predict the post-blended result from
        the raw features alone — making future inferences more accurate.
        """
        if not self.is_training or not self.available:
            return

        try:
            feat_vec = [float(features.get(k, 0.0)) for k in _FEATURE_KEYS]
            feat_tensor = _torch.tensor([feat_vec], dtype=_torch.float32)

            emotions = list(target_probs.keys())
            target_vec = [float(target_probs.get(e, 0.0)) for e in emotions]
            target_tensor = _torch.tensor([target_vec], dtype=_torch.float32)

            self._model.train()
            pred = self._model(feat_tensor).squeeze(0)

            # Convert logits to probabilities for loss computation
            pred_probs = _torch.softmax(pred, dim=0)
            loss = _nn.functional.kl_div(
                _torch.log(pred_probs + 1e-8),
                target_tensor,
                reduction="sum",
            )

            self._optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            _torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            self._model.eval()

            self._step_count += 1
            self._cumulative_loss += float(loss.item())

            # Periodic save (every 100 steps)
            if self._step_count % 100 == 0:
                self._save_weights()

        except Exception as exc:
            self._log(f"[NeuralRefiner] Online learning step failed: {exc}")

    def _load_weights(self):
        """Load saved model weights if they exist."""
        if not self.available:
            return
        try:
            if _NEURAL_REFINER_PATH.exists():
                state = _torch.load(str(_NEURAL_REFINER_PATH), map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
                self._model.eval()
                if _NEURAL_REFINER_META.exists():
                    meta = json.loads(_NEURAL_REFINER_META.read_text(encoding="utf-8"))
                    self._step_count = meta.get("step_count", 0)
                    self._cumulative_loss = meta.get("cumulative_loss", 0.0)
                self._log(f"[NeuralRefiner] Loaded weights from {_NEURAL_REFINER_PATH.name} (step {self._step_count})")
        except Exception as exc:
            self._log(f"[NeuralRefiner] Could not load weights: {exc} — starting fresh")

    def _save_weights(self):
        """Persist model weights to disk."""
        if not self.available:
            return
        try:
            _NEURAL_REFINER_DIR.mkdir(parents=True, exist_ok=True)
            _torch.save(self._model.state_dict(), str(_NEURAL_REFINER_PATH))
            meta = {
                "step_count": self._step_count,
                "cumulative_loss": round(self._cumulative_loss, 4),
                "saved_at": time.time(),
            }
            _NEURAL_REFINER_META.write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            self._log(f"[NeuralRefiner] Could not save weights: {exc}")

    def status(self) -> dict[str, Any]:
        """Return a status dict for the system tab / telemetry."""
        return {
            "available": self.available,
            "training_active": self.is_training,
            "step_count": self._step_count,
            "last_inference_ms": round(self._last_inference_ms, 2),
            "avg_loss": round(self._cumulative_loss / max(1, self._step_count), 4),
            "refinement_enabled": self._refinement_enabled,
        }

    def shutdown(self):
        """Save weights on shutdown."""
        if self.available and self._step_count > 0:
            self._save_weights()
