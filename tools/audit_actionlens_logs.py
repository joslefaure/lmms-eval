#!/usr/bin/env python3
"""
Audit script for ActionLens evaluation logs across all 20 evaluated VLMs
(18 full-benchmark runs of 6,701 items + 8 human-review subset runs of 1,194 items).

Usage:
    python tools/audit_actionlens_logs.py [--outputs-dir outputs/actionlens]
"""

import argparse
import glob
import hashlib
import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TASKS = [
    ("D1", "d1_transition_sensitivity", 194, 194),
    ("D2", "d2_actor_action_disambiguation", 2000, 250),
    ("D3", "d3_concurrent_action_binding", 2000, 250),
    ("D4", "d4_directed_interaction_reasoning", 1611, 250),
    ("D5", "d5_gaze_direction_attribution", 896, 250),
]

# Paper Table 2 numbers: [D1, D2, D3, D4, D5, Avg]
PAPER_TABLE2_SUBSET = [
    ("GPT-5.2", "gpt-5.2", [58.8, 76.0, 76.4, 70.0, 47.2, 66.0]),
    ("Gemini 3 Flash", "gemini-3-flash-preview", [65.5, 62.8, 73.6, 59.2, 46.0, 61.2]),
    ("Gemma-4-31B", "google__gemma-4-31B-it", [63.9, 66.0, 74.4, 61.6, 46.0, 62.3]),
    ("LLaVA-Video-32B", "lmms-lab__LLaVA-NeXT-Video-32B-Qwen", [60.3, 47.2, 40.8, 47.6, 28.4, 44.1]),
    ("InternVL3.5-38B", "OpenGVLab__InternVL3_5-38B", [68.0, 72.8, 67.6, 68.4, 31.2, 61.3]),
    ("Qwen3.5-27B", "Qwen__Qwen3.5-27B", [64.9, 70.0, 78.4, 76.4, 39.6, 65.9]),
    ("InternVL3.5-4B", "OpenGVLab__InternVL3_5-4B", [61.9, 60.4, 68.4, 46.8, 23.6, 51.8]),
    ("Qwen3.5-4B", "Qwen__Qwen3.5-4B", [60.3, 67.2, 74.0, 66.8, 33.6, 60.4]),
]

PAPER_TABLE2_FULL = [
    ("LLaVA-Video-32B", "lmms-lab__LLaVA-NeXT-Video-32B-Qwen", [60.3, 45.0, 42.0, 51.0, 28.7, 43.8]),
    ("Qwen3.5-27B", "Qwen__Qwen3.5-27B", [64.9, 67.7, 74.8, 77.3, 43.3, 68.8]),
    ("Qwen3-VL-32B", "Qwen__Qwen3-VL-32B-Instruct", [63.4, 64.0, 66.9, 68.8, 40.5, 62.8]),
    ("Gemma-4-31B", "google__gemma-4-31B-it", [63.9, 63.0, 72.9, 61.6, 42.0, 62.8]),
    ("Qwen2.5-VL-32B", "Qwen__Qwen2.5-VL-32B-Instruct", [66.0, 54.4, 70.5, 64.7, 34.3, 59.3]),
    ("Gemma-3-27B", "google__gemma-3-27b-it", [67.0, 37.9, 35.1, 26.4, 25.8, 33.5]),
    ("InternVL3.5-38B", "OpenGVLab__InternVL3_5-38B", [68.0, 65.5, 68.4, 68.7, 38.6, 63.6]),
    ("Qwen3.5-9B", "Qwen__Qwen3.5-9B", [64.9, 64.6, 69.1, 70.3, 37.6, 63.7]),
    ("Qwen3-VL-8B", "Qwen__Qwen3-VL-8B-Instruct", [61.9, 57.4, 70.9, 66.2, 30.8, 60.1]),
    ("InternVL3.5-8B", "OpenGVLab__InternVL3_5-8B", [60.3, 58.7, 65.5, 59.7, 33.6, 57.6]),
    ("VideoLLaMA3-7B", "DAMO-NLP-SG__VideoLLaMA3-7B", [71.6, 51.7, 52.6, 58.4, 27.3, 50.9]),
    ("LLaVA-OneVision-7B", "lmms-lab__llava-onevision-qwen2-7b-ov", [67.0, 47.5, 53.8, 53.4, 27.8, 48.7]),
    ("LLaVA-Video-7B", "lmms-lab__LLaVA-NeXT-Video-7B-DPO", [53.6, 34.0, 53.9, 39.2, 25.0, 40.6]),
    ("Qwen3.5-4B", "Qwen__Qwen3.5-4B", [60.3, 66.0, 69.9, 69.0, 35.3, 63.6]),
    ("Qwen3-VL-4B", "Qwen__Qwen3-VL-4B-Instruct", [68.0, 60.0, 67.6, 63.4, 30.1, 59.3]),
    ("InternVL3.5-4B", "OpenGVLab__InternVL3_5-4B", [61.9, 54.8, 66.2, 52.5, 31.6, 54.7]),
    ("Gemma-4-E4B", "google__gemma-4-E4B-it", [52.1, 47.9, 64.6, 43.2, 28.2, 49.2]),
    ("Gemma-3-4B", "google__gemma-3-4b-it", [48.5, 30.7, 27.2, 24.3, 24.2, 27.7]),
]


def round_half_up(pct: float) -> float:
    return float(Decimal(str(round(pct, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def round_half_even(pct: float) -> float:
    return round(round(pct, 6), 1)


def classify_and_extract_stated_letter(response: str) -> Tuple[Optional[str], str]:
    """
    Classify raw model response and extract its explicitly stated option letter.
    Returns (letter_or_None, category) where category is one of:
      - "leading_option": starts directly with A/B/C/D (e.g. "A", "A.", "A. walking")
      - "stated_in_text": states choice in Markdown/prose (e.g. "The correct answer is **A**.")
      - "no_clear_choice": refusal or truncated prose with no unambiguous choice
    """
    s = (response or "").strip()
    if re.match(r"^[A-D](?:[\.\)\s,:]|$)", s):
        return s[0].upper(), "leading_option"

    patterns = [
        r"(?:correct answer is|answer is|Answer:|option is|choose|select)\s*\*\*([A-D])\b",
        r"\*\*([A-D])[\.\)\*]",
        r"\*\*([A-D])\*\*",
        r"(?:correct answer is|answer is|Answer:|option is)\s*([A-D])\b",
        r"\b([A-D])\s+is\s+(?:the\s+)?correct",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).upper(), "stated_in_text"

    return None, "no_clear_choice"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_single_run(
    outputs_dir: Path,
    scope: str,
    paper_name: str,
    model_dir: str,
    paper_scores: List[float],
) -> Dict:
    run_dir = outputs_dir / scope / model_dir
    res_files = sorted(run_dir.glob("*_results.json"))
    if not res_files:
        raise FileNotFoundError(f"No *_results.json found in {run_dir}")
    res_path = res_files[-1]
    prefix = str(res_path)[: -len("_results.json")]

    per_task = {}
    tot_n = 0
    tot_saved_correct = 0
    tot_stated_correct = 0
    tot_non_leading = 0
    tot_no_clear_choice = 0

    for diag_id, task_key, exp_full_n, exp_sub_n in TASKS:
        expected_n = exp_full_n if scope == "full" else exp_sub_n
        sample_path = Path(f"{prefix}_samples_{task_key}.jsonl")
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample log: {sample_path}")

        n = 0
        saved_correct = 0
        stated_correct = 0
        non_leading = 0
        no_clear = 0

        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                info = row["actionlens_accuracy"]
                n += 1
                gt = info.get("expected") or info.get("gold") or row.get("target")
                resp = info.get("response") if "response" in info else info.get("pred_raw", "")
                saved_c = int(info["correct"])
                saved_correct += saved_c

                stated_pred, kind = classify_and_extract_stated_letter(resp)
                if kind != "leading_option":
                    non_leading += 1
                if kind == "no_clear_choice":
                    no_clear += 1
                if stated_pred == gt:
                    stated_correct += 1

        if n != expected_n:
            raise ValueError(f"{sample_path}: expected N={expected_n}, found N={n}")

        per_task[diag_id] = {
            "task_key": task_key,
            "file": str(sample_path.relative_to(outputs_dir.parent.parent)),
            "sha256": sha256_file(sample_path),
            "n": n,
            "saved_correct": saved_correct,
            "saved_acc_pct": round_half_up(saved_correct / n * 100),
            "stated_correct": stated_correct,
            "stated_acc_pct": round_half_up(stated_correct / n * 100),
            "non_leading_option": non_leading,
            "no_clear_choice": no_clear,
        }
        tot_n += n
        tot_saved_correct += saved_correct
        tot_stated_correct += stated_correct
        tot_non_leading += non_leading
        tot_no_clear_choice += no_clear

    use_stated = paper_name == "Gemini 3 Flash"
    active_corrects = [
        per_task[d]["stated_correct" if use_stated else "saved_correct"]
        for d, _, _, _ in TASKS
    ] + [tot_stated_correct if use_stated else tot_saved_correct]
    active_ns = [per_task[d]["n"] for d, _, _, _ in TASKS] + [tot_n]
    active_pcts = [c / n * 100 for c, n in zip(active_corrects, active_ns)]

    col_names = ["D1", "D2", "D3", "D4", "D5", "Avg"]
    discrepancies = []
    for col, p_val, raw_pct, c, n in zip(col_names, paper_scores, active_pcts, active_corrects, active_ns):
        hu = round_half_up(raw_pct)
        he = round_half_even(raw_pct)
        if p_val != hu and p_val != he:
            discrepancies.append({
                "column": col,
                "paper_pct": p_val,
                "log_pct": hu,
                "correct": c,
                "n": n,
            })

    return {
        "scope": scope,
        "model": paper_name,
        "model_dir": model_dir,
        "results_file": str(res_path.relative_to(outputs_dir.parent.parent)),
        "results_sha256": sha256_file(res_path),
        "n_total": tot_n,
        "saved_correct_total": tot_saved_correct,
        "saved_avg_pct": round_half_up(tot_saved_correct / tot_n * 100),
        "stated_correct_total": tot_stated_correct,
        "stated_avg_pct": round_half_up(tot_stated_correct / tot_n * 100),
        "non_leading_total": tot_non_leading,
        "no_clear_choice_total": tot_no_clear_choice,
        "paper_table2_scores": paper_scores,
        "verified_log_scores": [round_half_up(p) for p in active_pcts],
        "discrepancies_vs_paper": discrepancies,
        "diagnostics": per_task,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit ActionLens logs for all 20 models.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "actionlens",
        help="Path to outputs/actionlens directory",
    )
    args = parser.parse_args()
    outputs_dir = args.outputs_dir.resolve()

    all_audits = []
    print("=" * 96)
    print("ACTIONLENS 20-MODEL PER-SAMPLE LOG AUDIT (18 Full-Set Runs + 8 Review-Subset Runs)")
    print("=" * 96)

    for scope, specs in [("subset", PAPER_TABLE2_SUBSET), ("full", PAPER_TABLE2_FULL)]:
        print(f"\n[{scope.upper()} EVALUATION ({1194 if scope == 'subset' else 6701} items)]")
        print(f"{'Model':20s} | {'N':>5s} | {'NonLead':>7s} | {'NoChoice':>8s} | {'D1':>5s} {'D2':>5s} {'D3':>5s} {'D4':>5s} {'D5':>5s} {'Avg':>5s} | Status")
        print("-" * 96)
        for paper_name, model_dir, paper_scores in specs:
            rec = audit_single_run(outputs_dir, scope, paper_name, model_dir, paper_scores)
            all_audits.append(rec)
            s = rec["verified_log_scores"]
            if not rec["discrepancies_vs_paper"]:
                status = "EXACT MATCH"
            else:
                parts = [f"{d['column']}: log={d['log_pct']}% ({d['correct']}/{d['n']}) vs paper={d['paper_pct']}%" for d in rec["discrepancies_vs_paper"]]
                status = "DIFF (" + "; ".join(parts) + ")"
            print(
                f"{paper_name:20s} | {rec['n_total']:5d} | {rec['non_leading_total']:7d} | "
                f"{rec['no_clear_choice_total']:8d} | {s[0]:5.1f} {s[1]:5.1f} {s[2]:5.1f} "
                f"{s[3]:5.1f} {s[4]:5.1f} {s[5]:5.1f} | {status}"
            )

    summary_path = outputs_dir / "audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_audits}, f, indent=2)

    report_path = outputs_dir / "AUDIT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ActionLens Per-Sample Log Audit (All 20 Models)\n\n")
        f.write("Reproduce this audit with:\n```bash\npython tools/audit_actionlens_logs.py\n```\n\n")
        for scope, title in [("subset", "Review Subset (1,194 items)"), ("full", "Full Benchmark (6,701 items)")]:
            f.write(f"## {title}\n\n")
            f.write("| Model | N | Non-Leading | No Clear Choice | D1 (%) | D2 (%) | D3 (%) | D4 (%) | D5 (%) | Avg. (%) | Paper Match |\n")
            f.write("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|\n")
            for rec in [r for r in all_audits if r["scope"] == scope]:
                s = rec["verified_log_scores"]
                if not rec["discrepancies_vs_paper"]:
                    match_str = "Exact match"
                else:
                    match_str = "; ".join(
                        f"{d['column']}: log `{d['log_pct']}%` (`{d['correct']}/{d['n']}`) vs. paper `{d['paper_pct']}%`"
                        for d in rec["discrepancies_vs_paper"]
                    )
                f.write(
                    f"| {rec['model']} | {rec['n_total']} | {rec['non_leading_total']} | "
                    f"{rec['no_clear_choice_total']} | {s[0]:.1f} | {s[1]:.1f} | {s[2]:.1f} | "
                    f"{s[3]:.1f} | {s[4]:.1f} | {s[5]:.1f} | {match_str} |\n"
                )
            f.write("\n")

    print(f"\nSaved JSON audit to {summary_path}")
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
