# MODIFIED: added relative path configuration
from pathlib import Path
ROOT = Path(__file__).resolve().parent
TARGET_VAR = "ret_crsp"
BASE_PATH = str(ROOT / "data_for_kit.csv")
LTV_PATH = str(ROOT / "LTV_History.csv")
VIX_PATH = str(ROOT / "VIX_History.csv")
INTRADAY_PATH = str(ROOT / "returns_5m.csv")