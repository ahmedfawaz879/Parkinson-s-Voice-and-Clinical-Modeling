"""Thin CLI wrapper so the pipeline can be run without installing the package.

    python main.py train --updrs
    python main.py robustness
    python main.py --help

Equivalent to the ``parkinsons-voice`` console script installed by
``pip install -e .`` (see pyproject.toml [project.scripts]).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from parkinsons_voice.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
