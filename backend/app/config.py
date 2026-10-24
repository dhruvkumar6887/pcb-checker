import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / 'config.yaml').read_text())
DATA = Path(os.environ.get('PCB_DATA_DIR', str(ROOT / 'data'))).resolve()
DATA.mkdir(parents=True, exist_ok=True)
