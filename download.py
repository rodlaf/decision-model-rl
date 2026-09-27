"""Fetch the pinned base and trained adapters. No token needed for public weights."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
BASE = 'AlexWortega/openjev'
BASE_REVISION = '552759daad712f1af6c4c13dabcb1e047886fc9c'
SUBFOLDER = 'qwen3.5-0.8b-nli-v5'
MODEL = 'rodneyslafuente/decision-model-rl-overcooked'
MODEL_REVISION = '5a704a53b6656db42b6b2e8ac4a6cf8dba2828b6'

if __name__ == '__main__':
    snapshot_download(BASE, revision=BASE_REVISION, allow_patterns=f'{SUBFOLDER}/*', local_dir=ROOT/'models/base')
    snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=ROOT/'models/trained')
