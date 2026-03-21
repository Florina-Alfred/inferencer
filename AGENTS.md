AGENTS

- Purpose: onboarding guide for agentic coding agents that operate on this repo. Use these commands and conventions when running, testing, linting or modifying code.

- Location: repository root (this file).


 - Running & Development
 - - Create a virtualenv and install dependencies (recommended): `uv sync` — this creates `.venv` and installs locked dependencies; then `source .venv/bin/activate` to enter the venv.
 - - If you change dependencies: run `uv lock` to regenerate `uv.lock`, then `uv sync` on other machines to apply the lock.
 - - Quick run: server and client are plain Python scripts; use `uv run` to execute them so the locked venv is used.
 - - - Start the UDP inference server: `uv run udp_infer_server.py`
 - - - Run the example live demo / single-process main: `uv run main.py -- --device cpu --model l --source 0`
 - - - Run the inference worker directly (useful for debugging):
 - -   - From python (inside the venv): `python -c "from inference_worker import run_inference_stream; run_inference_stream('0','127.0.0.1',56060)"`
 - -   - Or use the run wrapper: `uv run run_worker.py 0 127.0.0.1 56060`


 - Tests
 - - This repository contains one helper/test script `test_udp_client.py` that is not a pytest-style unit test; it is a runnable client you use to exercise the server.
 - - Run the script (single "test"): `uv run test_udp_client.py [SOURCE]`
 - -   - Example: `uv run test_udp_client.py 0` will send START/HEARTBEAT/STOP messages for source `0` to the server running on `127.0.0.1:55055` and listen on client port `56060`.
- - If you want to run tests with pytest you can do the following (recommended for CI/unit tests):
-   1) Convert or wrap `test_udp_client.py` logic into proper pytest tests (create `tests/` and functions prefixed with `test_`).
-   2) Run all tests: `pytest -q`
-   3) Run a single pytest test: `pytest -q tests/test_file.py::test_function_name -k "pattern"`
- - Tips for single-file/script debugging: use `PYTHONPATH=.` when running from another working directory, or `python -m test_udp_client` if you add a package wrapper.


 - Build / Packaging
 - - This project uses a minimal `pyproject.toml` and `setup.py` exists. For developer installs use the locked environment:
 - -   - `uv sync` to create and install the venv with pinned deps. If you explicitly need an editable install, run `pip install -e .` inside the activated venv.
 - - - For building sdist/wheel: `python -m build` (requires the `build` package) then `twine upload dist/*` to publish.


 - Linting & Formatting (recommended toolchain)
 - - Formatting: `black` (line-length 88). Run: `black .`
 - - Linting & autofix: `ruff` — a fast linter/formatter/autofixer that can replace `isort`/`flake8` workflows. Run: `ruff check .` and auto-fix with `ruff format --fix .`.
 - - Import sorting & minor fixes: configure `ruff` in `pyproject.toml` to handle import sorting and rule sets consistently across the repo.
 - - Static typing: `mypy .` (recommend enabling incremental mode in CI)
 - - Run a focused lint pass on one file: `black path/to/file.py && ruff check path/to/file.py`
 - - Pre-commit: strongly recommend installing `pre-commit` and enabling hooks inside the venv. Add `ruff` and `black` hooks to `.pre-commit-config.yaml` and run `pre-commit install && pre-commit run --all-files`.


- Code Style Guidelines
- - Formatting
-   - Use `black` as the source-of-truth formatter. Do not hand-format beyond what black enforces.
-   - Keep line length <= 88 characters unless there is a strong, justified reason.
-   - Use 4-space indentation (Python default).
- - Imports
-   - Group imports in three sections, each separated by a blank line: standard library, third-party, local application imports.
-   - Use absolute imports from the repository root when referring to local modules (e.g. `from utils import demo_postprocess`) — current codebase uses local imports by filename.
-   - Sort imports with `isort` (use `--profile black`).
- - Naming
-   - Modules / files: short, lower_snake_case (already used: `utils.py`, `main.py`).
-   - Functions and variables: lower_snake_case.
-   - Classes: PascalCase.
-   - Constants: UPPER_SNAKE_CASE (e.g. `LISTEN_PORT`, `IMAGE_SIZE`).
- - Types & Annotations
-   - Prefer adding type hints for public API functions and library code. For scripts and quick CLI glue code, types are optional but recommended.
-   - Example signature style: `def run_inference_stream(source: str, client_addr: str, client_port: int, model_name: str = "l") -> None:`
-   - Use `typing` primitives (`List`, `Tuple`, `Dict`) or `collections.abc` when needed. Keep annotations readable.
- - Error Handling
-   - Do not swallow exceptions silently. Use logging (`loguru` is used in this repo) to emit context and re-raise or return a controlled error state.
-   - Use specific exception types where possible (e.g. `except ValueError:` not `except Exception:`), and only catch `Exception` at top-level boundaries to keep process alive.
-   - Prefer `try/except/finally` when resources need explicit cleanup (sockets, video capture). The codebase already calls `cap.release()` and `sock.close()` in `finally` blocks — maintain this pattern.
- - Logging
-   - Use `loguru` consistently for library and service-level logging; fall back to `print()` only in quick scripts or during bootstrap/early startup.
-   - Emit structured logs when possible, e.g. `logger.info(json.dumps({log_key: detections}))` or `logger.bind(source=source).info("...")`.
- - Concurrency / Processes
-   - When spawning processes use `multiprocessing.Process` (existing code does this). Ensure you keep strong ownership over child processes and terminate/join them on shutdown.
-   - Protect main entry points with `if __name__ == "__main__":` to avoid accidental process forking on import.


- Testing Conventions
- - Unit tests: place under `tests/` and use pytest style functions. Name files `test_*.py` and functions `test_*`.
- - Integration/E2E: place under `tests/integration/` and mark with `@pytest.mark.integration` to separate slow network/process tests.
- - Use fixtures for ephemeral sockets and ports (pytest's `tmp_path` and `monkeypatch` are useful).
- - When testing networked components (UDP server/clients), prefer binding to `127.0.0.1` and ephemeral ports. Clean up processes and sockets in `finally` blocks.


- Misc / Repository Conventions
- - .gitignore already excludes `.venv` and `*.onnx` model artifacts. Do not check in large binary models into git.
- - Keep secrets out of repo. `.env` may contain local environment overrides — do not commit secrets.


- Cursor / Copilot Rules
- - Cursor rules: none found in `.cursor/rules/` or `.cursorrules` in this repository.
- - GitHub Copilot instructions: none found at `.github/copilot-instructions.md`.


 - Recommended Next Actions For Agents
 - 1) Run `uv run udp_infer_server.py` in one shell and `uv run test_udp_client.py 0` in another to exercise the end-to-end UDP flow.
 - 2) Add a `tests/` directory and convert `test_udp_client.py` to a pytest-based integration test (wrap process lifecycle in fixtures).
 - 3) Add `pyproject.toml` tool configs for `black`, `isort`, `mypy`, and `flake8` to ensure CI formatting and type checks are reproducible.


- Contact / Maintainer Notes
- - This project uses `loguru` and `onnxruntime-gpu` — if you run on a machine without CUDA, use `--device cpu` or ensure the fallback to CPUExecutionProvider works.
