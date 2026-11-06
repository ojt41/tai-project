#!/bin/bash
# Benign baseline + both attack sets on the reference setup (V100 16 GB, fp16).
# Submit from the repo root on Triton:  mkdir -p logs && sbatch rag/jobs/run_all.sh
# Re-submitting resumes: items already in results/v100-3b.jsonl are skipped.
#SBATCH --job-name=rag-run-all
#SBATCH --partition=gpu-v100-16g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

module load model-huggingface
uv run python -m rag.run_experiment --model Qwen/Qwen2.5-3B-Instruct --run-name v100-3b
