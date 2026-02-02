#!/usr/bin/env python3
"""
Generate gRPC Python stubs for edge/proto/infer.proto using grpc_tools.
Run from the edge/ directory or via the Makefile target `make gen`.
"""
import os
import sys

def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proto_dir = os.path.join(this_dir, "proto")
    proto_file = os.path.join(proto_dir, "infer.proto")
    if not os.path.exists(proto_file):
        print(f"proto file not found: {proto_file}", file=sys.stderr)
        sys.exit(1)

    # Prefer grpc_tools.protoc when available (no system protoc required)
    try:
        from grpc_tools import protoc as _protoc

        res = _protoc.main([
            "protoc",
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            proto_file,
        ])
    except Exception:
        # fallback to system 'protoc' if available
        import shutil
        import subprocess

        protoc_bin = shutil.which("protoc")
        if protoc_bin is None:
            print(
                "Neither grpc_tools.protoc nor system protoc found. Install grpcio-tools or protoc.",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd = [
            protoc_bin,
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            proto_file,
        ]
        proc = subprocess.run(cmd)
        res = proc.returncode

    if res != 0:
        print("protoc returned non-zero status", file=sys.stderr)
        sys.exit(res)
    # Post-process generated grpc files to ensure package-relative imports.
    # Some protoc/grpc_tools variants emit bare imports like
    #   import infer_pb2 as infer__pb2
    # which break when the files are imported as part of the `edge.proto`
    # package. Fix these occurrences in-place. This is idempotent: if the
    # file already uses a package-relative import nothing is changed.
    try:
        import re

        for name in os.listdir(proto_dir):
            if not name.endswith("_pb2_grpc.py"):
                continue
            path = os.path.join(proto_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            # replace bare imports like: "import infer_pb2 as infer__pb2"
            patched = re.sub(r"^import\s+(\w+_pb2)\s+as\s+(\w+)", r"from . import \1 as \2", src, flags=re.M)
            # replace simple bare imports: "import infer_pb2" -> "from . import infer_pb2"
            patched = re.sub(r"^import\s+(\w+_pb2)\s*$", r"from . import \1", patched, flags=re.M)
            if patched != src:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(patched)
                print(f"Patched package-relative imports in {path}")
    except Exception as e:
        print("Warning: failed to post-process generated stubs:", e, file=sys.stderr)

    print("Generated stubs in", proto_dir)

if __name__ == "__main__":
    main()
