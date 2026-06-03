"""
Console-script entry points for `pip install pss`.

The actual CLI logic lives in the example scripts at the repo root (under
`examples/`). Those scripts are designed to be runnable in-place
(`python examples/load_and_reduce.py ...`) without installation. When the
package is pip-installed, this module re-exposes them as the console scripts
declared in pyproject.toml:

    pss-load-reduce         <- examples/load_and_reduce.py
    pss-load-reduce-median  <- examples/load_and_reduce_with_median_gain.py
    pss-eta-demo            <- eta_field_recon/demo_eta_field.py

We bridge by adding the script's directory to sys.path on demand and
importing its main() function. This avoids duplicating CLI logic and means
edits to the scripts at the repo root automatically flow through to the
console-script behavior.
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
            f"could not locate console-script source at {script_path}; "
            f"if you installed from a wheel, the auxiliary scripts may "
            f"not have been bundled. Run the package modules directly "
            f"instead (e.g. `python examples/load_and_reduce.py`)."
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
    return _import_main_from("examples/load_and_reduce.py")(argv)


def load_and_reduce_median_gain_main(argv: list[str] | None = None) -> int:
    return _import_main_from("examples/load_and_reduce_with_median_gain.py")(argv)


def eta_demo_main(argv: list[str] | None = None) -> int:
    """Run the eta-reconstruction demo. Returns the int exit code of the
    demo script (which uses `if __name__ == '__main__'` rather than a
    `main()` function, so we shell out via runpy)."""
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
