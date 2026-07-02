"""
Console-script entry points for the `epss` distribution.

The actual CLI logic lives in the example scripts at the repo root
(`_examples/`), which are runnable in-place (`python
_examples/load_and_reduce.py ...`). This module re-exposes them as the
console scripts declared in pyproject.toml:

    pss-load-reduce         <- _examples/load_and_reduce.py
    pss-load-reduce-median  <- _examples/load_and_reduce_with_median_gain.py
    pss-skyaware-demo       <- _examples/skyaware_demo.py
    pss-eta-demo            <- eta_field_recon/demo_eta_field.py

CLONE/EDITABLE INSTALLS ONLY: `_examples/` and `_data/` are deliberately
excluded from wheels (the data module caches multi-GB Zenodo downloads
beside itself), so these scripts require the repository on disk
(`pip install -e .` from a clone). A wheel install raises the actionable
RuntimeError below.

The bridge adds the script's directory to sys.path on demand and imports its
main() function, so edits to the repo-root scripts flow through without
duplicated CLI logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_main_from(script_relpath: str, func_name: str = "main"):
    """Import a `main` function from a script that lives outside the package."""
    pkg_dir = Path(__file__).resolve().parent       # .../pss/
    repo_root = pkg_dir.parent                       # the Python package root
    script_path = repo_root / script_relpath
    if not script_path.exists():
        raise RuntimeError(
            f"could not locate console-script source at {script_path}. "
            f"The console scripts need the repository on disk (wheel installs "
            f"exclude _examples/ and _data/): clone the repo and install "
            f"editable (`pip install -e .`), then rerun, or run the script "
            f"directly (`python _examples/load_and_reduce.py`)."
        )
    spec = importlib.util.spec_from_file_location(
        f"_pss_cli_{script_path.stem}", script_path,
    )
    module = importlib.util.module_from_spec(spec)
    # Make sure the script can import `pss` even when called via console-script
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def load_and_reduce_main(argv: list[str] | None = None) -> int:
    return _import_main_from("_examples/load_and_reduce.py")(argv)


def load_and_reduce_median_gain_main(argv: list[str] | None = None) -> int:
    return _import_main_from("_examples/load_and_reduce_with_median_gain.py")(argv)


def skyaware_demo_main(argv: list[str] | None = None) -> int:
    return _import_main_from("_examples/skyaware_demo.py")(argv)


def eta_demo_main(argv: list[str] | None = None) -> int:
    """Run the eta-reconstruction demo via runpy (the script uses
    `if __name__ == '__main__'` rather than a main() function). Always
    returns 0; a sys.exit inside the demo propagates as SystemExit."""
    import runpy
    pkg_dir = Path(__file__).resolve().parent       # .../pss/
    repo_root = pkg_dir.parent                      # repo root
    script = repo_root / "eta_field_recon" / "demo_eta_field.py"
    if not script.exists():
        raise RuntimeError(f"could not locate {script}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    # Convenience: `python -m pss._cli` runs the default reduction demo.
    raise SystemExit(load_and_reduce_main())
