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
    print("Generated stubs in", proto_dir)

if __name__ == "__main__":
    main()
