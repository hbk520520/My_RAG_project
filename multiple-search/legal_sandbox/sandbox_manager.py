"""Compatibility import for agent_tools.sandbox.manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_tools.sandbox import manager
sys.modules[__name__] = manager
