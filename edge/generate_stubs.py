#!/usr/bin/env python3
"""Generate gRPC Python stubs for edge/proto/infer.proto.

This script prefers using grpc_tools.protoc from the active Python
environment. It does not mutate generated files; it only invokes the
code generator.
"""
import os
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    proto_dir = os.path.join(here, "proto")
    proto_file = os.path.join(proto_dir, "infer.proto")
    if not os.path.exists(proto_file):
        print("proto file not found: ", proto_file, file=sys.stderr)
        sys.exit(1)

    try:
        from grpc_tools import protoc

        args = [
            "protoc",
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            proto_file,
        ]
        res = protoc.main(args)
    except Exception:
        # fallback to system protoc
        import shutil
        import subprocess

        protoc_bin = shutil.which("protoc")
        if protoc_bin is None:
            print("grpc_tools.protoc not installed and protoc not found", file=sys.stderr)
            sys.exit(1)
        cmd = [protoc_bin, f"-I{proto_dir}", f"--python_out={proto_dir}", f"--grpc_python_out={proto_dir}", proto_file]
        proc = subprocess.run(cmd)
        res = proc.returncode

    if res != 0:
        print("protoc returned non-zero status", file=sys.stderr)
        sys.exit(res)

    print("Generated stubs in:", proto_dir)


if __name__ == "__main__":
    main()
