"""Compatibility entry point; implementation lives in legal_agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    from legal_agent.demo import main
    main()
else:
    from legal_agent import compat
    sys.modules[__name__] = compat
