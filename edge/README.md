Edge gRPC demo (minimal)

This folder contains a minimal, easy-to-understand starting point for an
edge gRPC demo. The repository's preferred workflow is to use the `uv`
tooling described in the project root `AGENTS.md` to manage the virtualenv
and Python tooling.

To set up and generate stubs:

  1. uv add grpcio grpcio-tools protobuf && uv sync
  2. source .venv/bin/activate
  3. python generate_stubs.py

After generating stubs you can run the server and client:

  - Start server: python infer_server.py --host 0.0.0.0 --port 50051
  - Start client: python sensor.py --host 127.0.0.1 --port 50051 --source 0

This folder intentionally keeps tooling and code minimal. The generated
files under `edge/proto` (the *_pb2.py and *_pb2_grpc.py) must not be
manually edited.
