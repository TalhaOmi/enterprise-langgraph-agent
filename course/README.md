# Course solutions (modules 4 to 22)

Tested solution code for the interactive course. Each module adds files under `src/ent_agent/`.
Run from this folder:

    pip install -e .
    PYTHONPATH=src python -m pytest
    python run_m4.py    # RAG; run_m17.py, run_m18.py, run_m20.py, run_m22.py for later modules

The full, production-hardened project is in the repository root.

Note: this folder and the root project both define a package named `ent_agent`.
Use a separate virtual environment for each, so they do not shadow each other.
