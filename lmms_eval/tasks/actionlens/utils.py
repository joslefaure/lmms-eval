"""ActionLens task utilities for lmms-eval.

ActionLens is a diagnostic benchmark for fine-grained, temporally-grounded
video action understanding.  It covers five diagnostics:

  D1 — Transition Sensitivity      (state changes in a single actor)
  D2 — Actor Action Disambiguation  (which actor performed an action)
  D3 — Concurrent Action Binding    (simultaneous actions in the clip)
  D4 — Directed Interaction Reasoning (who acts on whom)
  D5 — Gaze Direction Attribution   (where a person is looking)

Dataset: https://huggingface.co/datasets/anonymous-dataset-submission/actionlens

Videos are fetched automatically from HuggingFace Hub on first use and cached
under $HF_HOME (default: ~/.cache/huggingface/).  No manual download needed.
"""

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

# ── HF repo ──────────────────────────────────────────────────────────────────

HF_REPO_ID = os.getenv("ACTIONLENS_HF_REPO", "anonymous-dataset-submission/actionlens")

# ── Video resolution ──────────────────────────────────────────────────────────

_hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))

# Allow pointing to a local clone of the dataset (skips all HF downloads).
_LOCAL_DIR = os.getenv("ACTIONLENS_DATA_DIR", "")


@lru_cache(maxsize=1)
def _hf_hub_download_fn():
    """Lazy import of hf_hub_download so the task loads even without the package."""
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download
    except ImportError:
        eval_logger.warning("[actionlens] huggingface_hub not installed; set ACTIONLENS_DATA_DIR to a local dataset clone.")
        return None


def _resolve_video(video_path: str) -> Optional[str]:
    """Return a local path for *video_path* (relative within the HF repo).

    Resolution order:
    1. ACTIONLENS_DATA_DIR env var (local dataset clone, fast)
    2. HF hub cache (already downloaded on a previous run)
    3. hf_hub_download (downloads on first use, cached by HF)
    """
    if not video_path:
        return None

    # 1. Local override
    if _LOCAL_DIR:
        candidate = os.path.join(_LOCAL_DIR, video_path)
        if os.path.exists(candidate):
            return candidate

    # 2. Check the standard HF hub cache layout before attempting a download.
    #    HF stores dataset files at:
    #    $HF_HOME/hub/datasets--<org>--<repo>/snapshots/<hash>/<path>
    hub_root = os.path.join(_hf_home, "hub")
    repo_slug = "datasets--" + HF_REPO_ID.replace("/", "--")
    snapshots_dir = os.path.join(hub_root, repo_slug, "snapshots")
    if os.path.isdir(snapshots_dir):
        for snap in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, snap, video_path)
            if os.path.exists(candidate):
                return candidate

    # 3. Download via huggingface_hub (idempotent — cached after first call).
    hf_download = _hf_hub_download_fn()
    if hf_download is None:
        eval_logger.error("[actionlens] Cannot download video '{}': huggingface_hub unavailable.", video_path)
        return None

    try:
        local = hf_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=video_path,
        )
        return local
    except Exception as exc:
        eval_logger.warning("[actionlens] Failed to download '{}': {}", video_path, exc)
        return None


# ── doc_to_* helpers ──────────────────────────────────────────────────────────


def actionlens_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """Return a list with the resolved local video path, or [] on failure."""
    video_path = str(doc.get("video_path", "")).strip()
    resolved = _resolve_video(video_path)
    if not resolved:
        eval_logger.warning(
            "[actionlens] Video not found for segment_id='{}' (path='{}').",
            doc.get("segment_id", "?"),
            video_path,
        )
        return []
    return [resolved]


def actionlens_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Format the MCQ prompt, optionally wrapping with pre/post prompts."""
    kwargs = lmms_eval_specific_kwargs or {}

    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "\nAnswer with the option letter only.")

    question = doc.get("question", "")
    options = doc.get("options", [])
    options_str = "\n".join(options)

    return f"{pre}{question}\n{options_str}{post}"


def actionlens_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter (A/B/C/D)."""
    return str(doc.get("answer", "")).strip().upper()


# ── Scoring ───────────────────────────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_letter(text: str) -> str:
    """Extract the first A/B/C/D answer letter from a model response."""
    if not isinstance(text, str):
        return ""
    # Strip chain-of-thought if present
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip().upper()
    # Direct single letter
    if text in {"A", "B", "C", "D"}:
        return text
    # "The answer is B." style
    m = _LETTER_RE.search(text)
    return m.group(1) if m else ""


def actionlens_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Score a single prediction and return a payload for aggregation."""
    pred_raw = results[0] if results else ""
    pred = _extract_letter(pred_raw)
    gold = actionlens_doc_to_target(doc)
    correct = int(pred == gold and gold != "")

    return {
        "actionlens_accuracy": {
            "segment_id": doc.get("segment_id", ""),
            "task": doc.get("task", ""),
            "pred": pred,
            "pred_raw": pred_raw,
            "gold": gold,
            "correct": correct,
        }
    }


def actionlens_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Compute overall accuracy as a percentage."""
    if not results:
        return 0.0
    return 100.0 * sum(r["correct"] for r in results) / len(results)


def get_answer_letter(doc: Dict[str, Any]) -> str:
    """Alias used by doc_to_target in YAML configs."""
    return actionlens_doc_to_target(doc)
