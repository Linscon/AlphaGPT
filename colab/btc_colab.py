"""Compatibility entry point; authoritative implementation lives in btc_mining.core."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc_mining import core
if __name__ == '__main__':
    core.main()
else:
    sys.modules[__name__] = core
