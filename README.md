# RSA Evaluation

### Math & Reasoning Benchmarks

Evaluate on **AIME-25**, **HMMT-25**, **Reasoning Gym**, and **SuperGPQA** using `eval_loop.py`.

Run with:

```bash
python eval_loop.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dataset data/<dataset_name>/train.parquet \
  --output ./eval/<dataset_name> \
  --loops 10 \
  --k 4 \
  --population 16
```

Replace `<dataset_name>` with one of:

* `aime25`
* `hmmt25`
* `rg_games`
* `rg_cognition`
* `supergpqa`

### LiveCodeBench

For **LiveCodeBench**, use `eval_code.py`.

```bash
python eval_code.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output ./eval/livecodebench \
  --loops 10 \
  --k 4 \
  --population 16
```