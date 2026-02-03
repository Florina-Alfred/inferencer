"""Centralized shim to prepare runtime for importing generated gRPC stubs.

Responsibilities:
- Ensure repository root is on sys.path so package imports work when files are
  executed as scripts.
- Apply a safe wrapper around protobuf runtime validator to avoid hard
  VersionError during development when stubs were generated with a different
  protoc version. This is a temporary developer convenience.
- Import the generated stubs from `edge.proto` and register a bare-name alias
  in sys.modules (so generated files that use `import infer_pb2` still work).

Import this module at the top of scripts that need the generated stubs.
"""
from __future__ import annotations

import os
import sys
import importlib


def _ensure_repo_root_on_path() -> None:
    # When a file under edge/ is executed directly (python edge/foo.py),
    # sys.path[0] becomes the edge/ directory which prevents importing the
    # `edge` package. Ensure the repository root is on sys.path so
    # `from edge.proto import ...` works regardless of invocation style.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _wrap_protobuf_validator() -> None:
    try:
        import google.protobuf.runtime_version as _pb_runtime  # type: ignore

        _orig_validate = getattr(_pb_runtime, "ValidateProtobufRuntimeVersion", None)

        def _safe_validate(domain, major, minor, patch, suffix, filename):
            try:
                if _orig_validate is not None:
                    return _orig_validate(domain, major, minor, patch, suffix, filename)
            except Exception:
                # ignore mismatches in development; warn via warnings if desired
                import warnings

                warnings.warn("Ignored protobuf runtime/gencode mismatch during import")
                return None

        setattr(_pb_runtime, "ValidateProtobufRuntimeVersion", _safe_validate)
    except Exception:
        # If protobuf runtime isn't available, let subsequent imports show useful errors
        return


def prepare_stubs() -> None:
    _ensure_repo_root_on_path()
    _wrap_protobuf_validator()

    try:
        infer_pb2 = importlib.import_module("edge.proto.infer_pb2")
        # register bare-name alias so generated grpc files using `import infer_pb2`
        # resolve correctly without modifying generated files.
        sys.modules.setdefault("infer_pb2", infer_pb2)
        importlib.import_module("edge.proto.infer_pb2_grpc")
    except Exception as exc:
        raise RuntimeError(
            "gRPC stubs not found in edge/proto. Run `cd edge && python generate_stubs.py`"
        ) from exc


# Run preparation on import for convenience
prepare_stubs()
