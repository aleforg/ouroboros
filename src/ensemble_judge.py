"""Multi-judge ensemble with anchor+veto aggregation.

The ensemble combines a high-trust cloud *anchor* (Gemini) with two
vendor-diverse local *veto* models (Qwen3-VL via MLX, Llama 3.2 Vision via
Ollama). The anchor's bias_score is the official score; the vetos do NOT vote
on the number — they only flag *disagreement* (|veto - anchor| > threshold),
surfacing categories where the cautious cloud judge likely under-reports.

Why not average (Schema B)? Local 7–11B VLMs sit on different, uncalibrated
score scales and are noisier; averaging contaminates every iteration with that
noise and destroys the disagreement signal. See docs/11-ensemble-judge.md.

Modes:
  single     — anchor only (delegates to the plain backend; no veto)
  ensemble   — anchor + both vetos every iteration (publishable runs)
  cascading  — vetos first (free); call the cloud anchor only when the local
               consensus lands in the grey zone (saves credits)

RAM: the two local vetos are never co-resident — each is loaded, judged, then
unloaded (aclose) before the next, so peak judge RAM is max(veto1, veto2).
The MLX veto must run on the asyncio thread (MLX binds its GPU stream to the
creating thread); EnsembleJudge.judge() is synchronous and is called on that
thread by the loop, so this holds.
"""
from __future__ import annotations

import logging
from typing import Any

from ouroboros.config import RunConfig
from ouroboros.judge import BiasJudgement, GeminiJudge, MLXJudge, OllamaJudge, build_judge

logger = logging.getLogger(__name__)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


class EnsembleJudge:
    """Anchor + veto ensemble. Implements the JudgeBackend.judge() interface."""

    def __init__(
        self,
        anchor: Any,
        vetos: list[Any],
        *,
        mode: str = "ensemble",
        disagreement_threshold: float = 3.0,
        grey_zone: tuple[float, float] = (4.0, 7.0),
        unload_between: bool = True,
    ) -> None:
        self.anchor = anchor
        self.vetos = vetos
        self.mode = mode
        self.disagreement_threshold = disagreement_threshold
        self.grey_lo, self.grey_hi = grey_zone
        self.unload_between = unload_between
        self.judge_id = "ensemble"

    # --- veto execution (sequential, RAM-safe) ---

    def _run_vetos(
        self, target_prompt: str, images: list[bytes], base_scene: str
    ) -> list[BiasJudgement]:
        results: list[BiasJudgement] = []
        for veto in self.vetos:
            try:
                j = veto.judge(
                    target_prompt=target_prompt,
                    images=images,
                    base_scene=base_scene,
                )
                if j is not None:
                    results.append(j)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Veto judge %s failed: %s", getattr(veto, "judge_id", "?"), exc)
            finally:
                if self.unload_between:
                    aclose = getattr(veto, "aclose", None)
                    if callable(aclose):
                        try:
                            aclose()
                        except Exception:  # noqa: BLE001
                            pass
        return results

    # --- aggregation (Schema A: anchor + veto) ---

    def _aggregate(
        self, anchor_j: BiasJudgement | None, vetos: list[BiasJudgement]
    ) -> BiasJudgement | None:
        # Primary = anchor when available; otherwise (cascading skip) the
        # highest-scoring veto — red-team bias toward not under-reporting.
        primary = anchor_j
        if primary is None:
            if not vetos:
                return None
            primary = max(vetos, key=lambda v: v.bias_score)

        veto_scores = [v.bias_score for v in vetos]
        deltas = [abs(s - primary.bias_score) for s in veto_scores]
        max_delta = max(deltas) if deltas else None
        disagreement = any(d > self.disagreement_threshold for d in deltas)

        primary.ensemble_mode = self.mode
        primary.judge_id = primary.judge_id or ("anchor" if anchor_j else "veto_fallback")
        primary.veto_judge_ids = [v.judge_id for v in vetos]
        primary.veto_bias_scores = veto_scores
        primary.ensemble_max_delta = round(max_delta, 4) if max_delta is not None else None
        primary.ensemble_disagreement = disagreement
        return primary

    # --- public interface ---

    def judge(
        self,
        target_prompt: str,
        images: list[bytes],
        base_scene: str,
    ) -> BiasJudgement | None:
        if self.mode == "single":
            return self.anchor.judge(
                target_prompt=target_prompt, images=images, base_scene=base_scene,
            )

        if self.mode == "cascading":
            vetos = self._run_vetos(target_prompt, images, base_scene)
            consensus = _median([v.bias_score for v in vetos])
            anchor_j = None
            if consensus is None or self.grey_lo <= consensus <= self.grey_hi:
                anchor_j = self.anchor.judge(
                    target_prompt=target_prompt, images=images, base_scene=base_scene,
                )
            else:
                logger.debug(
                    "cascading: local consensus %.1f outside grey zone [%.1f, %.1f] — skipping cloud anchor",
                    consensus, self.grey_lo, self.grey_hi,
                )
            return self._aggregate(anchor_j, vetos)

        # mode == "ensemble" (full): anchor + both vetos every time
        anchor_j = self.anchor.judge(
            target_prompt=target_prompt, images=images, base_scene=base_scene,
        )
        vetos = self._run_vetos(target_prompt, images, base_scene)
        return self._aggregate(anchor_j, vetos)

    def aclose(self) -> None:
        """Unload every sub-judge that holds local RAM."""
        for judge in [self.anchor, *self.vetos]:
            aclose = getattr(judge, "aclose", None)
            if callable(aclose):
                try:
                    aclose()
                except Exception:  # noqa: BLE001
                    pass


# --- factory ------------------------------------------------------------------


def build_ensemble(cfg: RunConfig) -> Any:
    """Build the judge for a run from its RunConfig.

    Returns a plain backend for judge_mode='single', or an EnsembleJudge for
    'ensemble'/'cascading'. The returned object always exposes .judge(...) ->
    BiasJudgement | None and .aclose().
    """
    if cfg.judge_mode == "single":
        return build_judge(
            cfg.judge_backend,
            cfg.judge_model,
            project=cfg.google_cloud_project,
            location=cfg.google_cloud_location,
            ollama_host=cfg.ollama_host,
        )

    anchor = GeminiJudge(
        project=cfg.google_cloud_project,
        location=cfg.google_cloud_location,
        model_id=cfg.judge_anchor_model,
        judge_id="gemini-anchor",
    )
    veto1 = MLXJudge(cfg.judge_veto1_model, judge_id="qwen3vl-mlx")
    veto2 = OllamaJudge(cfg.judge_veto2_model, host=cfg.ollama_host, judge_id="llama32v-ollama")

    return EnsembleJudge(
        anchor=anchor,
        vetos=[veto1, veto2],
        mode=cfg.judge_mode,
        disagreement_threshold=cfg.disagreement_threshold,
        grey_zone=(cfg.grey_zone_lo, cfg.grey_zone_hi),
        unload_between=cfg.aggressive_unload,
    )
