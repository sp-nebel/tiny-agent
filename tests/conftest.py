import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture(autouse=True)
def _no_global_instructions(monkeypatch, tmp_path_factory):
    """A real ~/.config/tiny-agent/AGENTS.md must not leak into first
    messages the tests assert on, nor its commands/ into the REPL."""
    import config
    monkeypatch.setattr(config, "GLOBAL_INSTRUCTIONS",
                        str(tmp_path_factory.getbasetemp() / "no-such-AGENTS.md"))
    # Nor the user's own custom commands (~/.config/tiny-agent/commands).
    monkeypatch.setattr(config, "CONFIG_DIR",
                        str(tmp_path_factory.getbasetemp() / "no-such-config"))
