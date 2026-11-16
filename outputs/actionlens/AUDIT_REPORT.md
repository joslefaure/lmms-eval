# ActionLens Per-Sample Log Audit (All 20 Models)

Reproduce this audit with:
```bash
python tools/audit_actionlens_logs.py
```

## Review Subset (1,194 items)

| Model | N | Non-Leading | No Clear Choice | D1 (%) | D2 (%) | D3 (%) | D4 (%) | D5 (%) | Avg. (%) | Paper Match |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| GPT-5.2 | 1194 | 0 | 0 | 58.8 | 76.0 | 76.4 | 70.0 | 47.2 | 66.0 | Exact match |
| Gemini 3 Flash | 1194 | 388 | 27 | 65.5 | 62.8 | 73.6 | 59.2 | 46.0 | 61.2 | Exact match |
| Gemma-4-31B | 1194 | 25 | 25 | 63.9 | 66.0 | 74.4 | 61.6 | 46.0 | 62.3 | Exact match |
| LLaVA-Video-32B | 1194 | 0 | 0 | 60.3 | 47.2 | 40.8 | 47.6 | 28.4 | 44.1 | Exact match |
| InternVL3.5-38B | 1194 | 0 | 0 | 68.0 | 72.8 | 67.6 | 68.4 | 31.2 | 61.3 | Exact match |
| Qwen3.5-27B | 1194 | 0 | 0 | 64.9 | 70.0 | 78.4 | 76.4 | 39.6 | 65.9 | Exact match |
| InternVL3.5-4B | 1194 | 0 | 0 | 61.9 | 60.4 | 68.4 | 46.8 | 23.6 | 51.8 | Exact match |
| Qwen3.5-4B | 1194 | 0 | 0 | 60.3 | 67.2 | 74.0 | 66.8 | 33.6 | 60.4 | Exact match |

## Full Benchmark (6,701 items)

| Model | N | Non-Leading | No Clear Choice | D1 (%) | D2 (%) | D3 (%) | D4 (%) | D5 (%) | Avg. (%) | Paper Match |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| LLaVA-Video-32B | 6701 | 0 | 0 | 60.3 | 45.0 | 42.0 | 51.0 | 28.7 | 43.8 | Exact match |
| Qwen3.5-27B | 6701 | 0 | 0 | 64.9 | 67.7 | 74.8 | 77.3 | 43.3 | 68.8 | Exact match |
| Qwen3-VL-32B | 6701 | 0 | 0 | 63.4 | 64.0 | 66.9 | 68.8 | 40.5 | 62.8 | Exact match |
| Gemma-4-31B | 6701 | 64 | 64 | 63.9 | 63.0 | 72.9 | 61.6 | 42.0 | 62.8 | Exact match |
| Qwen2.5-VL-32B | 6701 | 0 | 0 | 66.0 | 54.4 | 70.5 | 64.7 | 34.3 | 59.3 | Exact match |
| Gemma-3-27B | 6701 | 55 | 55 | 67.0 | 37.9 | 35.1 | 26.4 | 25.8 | 33.5 | Exact match |
| InternVL3.5-38B | 6701 | 0 | 0 | 68.0 | 65.5 | 68.4 | 68.7 | 38.6 | 63.6 | Exact match |
| Qwen3.5-9B | 6701 | 0 | 0 | 64.9 | 64.6 | 69.1 | 70.3 | 37.6 | 63.7 | Exact match |
| Qwen3-VL-8B | 6701 | 0 | 0 | 61.9 | 57.4 | 70.9 | 66.2 | 30.8 | 60.1 | Exact match |
| InternVL3.5-8B | 6701 | 0 | 0 | 60.3 | 58.7 | 65.5 | 59.7 | 33.6 | 57.6 | Exact match |
| VideoLLaMA3-7B | 6701 | 0 | 0 | 71.6 | 51.7 | 52.6 | 58.4 | 27.3 | 50.9 | Exact match |
| LLaVA-OneVision-7B | 6701 | 0 | 0 | 67.0 | 47.5 | 53.8 | 53.4 | 27.8 | 48.7 | Exact match |
| LLaVA-Video-7B | 6701 | 0 | 0 | 53.6 | 34.1 | 53.9 | 39.2 | 25.0 | 40.6 | Exact match |
| Qwen3.5-4B | 6701 | 0 | 0 | 60.3 | 66.0 | 69.9 | 69.0 | 35.3 | 63.6 | Exact match |
| Qwen3-VL-4B | 6701 | 0 | 0 | 68.0 | 60.1 | 67.6 | 63.4 | 30.1 | 59.3 | Exact match |
| InternVL3.5-4B | 6701 | 0 | 0 | 61.9 | 54.8 | 66.2 | 52.5 | 31.6 | 54.7 | Exact match |
| Gemma-4-E4B | 6701 | 0 | 0 | 52.1 | 47.9 | 64.6 | 43.2 | 28.2 | 49.2 | Exact match |
| Gemma-3-4B | 6701 | 37 | 37 | 48.5 | 30.7 | 27.2 | 24.3 | 24.2 | 27.7 | Exact match |

