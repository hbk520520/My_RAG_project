"""Compatibility entry point for legal_agent.routing."""
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("legal_agent.routing", run_name="__main__")
else:
    from legal_agent import routing
    sys.modules[__name__] = routing
