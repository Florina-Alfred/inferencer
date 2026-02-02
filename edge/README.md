Edge gRPC minimal demo
======================

What this folder contains
- `proto/infer.proto` — protobuf definition for a bidirectional Stream RPC
- `generate_stubs.py` & `Makefile` — convenience helper to generate Python gRPC stubs
- `infer_server.py` — minimal gRPC server that accepts base64 JPEG frames and streams detections
- `sensor.py` — minimal webcam client that streams frames and overlays detections

Design goals
- Minimal and easy to run (single-file server & client)
- Low CPU / network usage: frames are downscaled (default 320x240) and JPEG-compressed
- Uses gzip compression in gRPC to reduce bandwidth further
- Base64 is used for portability; switch to raw bytes if you need lower overhead

Quick start

1. Add runtime dependencies to the locked project environment and install them (use `uv`):

   ```bash
   # add packages to the project lock (updates pyproject/uv.lock)
   uv add grpcio grpcio-tools protobuf opencv-python-headless numpy pillow
   # install the locked venv (creates .venv and installs pinned deps)
   uv sync
   # activate the venv
   source .venv/bin/activate
   ```

2. Generate Python gRPC stubs (from the `edge/` directory):

   ```bash
   cd edge
   make gen
   # or: python generate_stubs.py
   ```

Why both a Python script and a Makefile?
- `generate_stubs.py` calls `grpc_tools.protoc` (if installed in the venv) or falls back to
  the system `protoc` binary. This makes generation portable across environments and avoids
  requiring a separate `protoc` install when you already have `grpcio-tools` in your venv.
- The `Makefile` is a tiny convenience wrapper so you can run `make gen` and keep the
  command short. Both exist so you can choose the workflow you prefer.

3. Run the server (example listens on 0.0.0.0:50051):

   ```bash
   python infer_server.py --host 0.0.0.0 --port 50051
   ```

4. Run the sensor (client) locally and point it at the server:

   ```bash
   python sensor.py --host 127.0.0.1 --port 50051 --source 0
   # press 'q' to quit the display
   ```

Notes & configuration
- By default the client resizes frames to 320x240 and encodes JPEG at quality 70. To change:
  - `sensor.py --width 640 --height 480 --quality 80`
- The server uses a placeholder detector that returns a centered box. Replace the placeholder
  logic in `infer_server.py` with a real inference function (for example one that calls
  `inference_worker.py` or an ONNX runtime wrapper) that accepts a numpy array and returns
  detections in normalized coordinates.
- If you want to avoid base64 overhead, change `image_b64` in the proto to `bytes image` and
  update client/server encode/decode accordingly. That will be more efficient on the wire.
- Both server and client set gRPC max message sizes to 50MB — adjust in the channel/server
  options if you expect larger frames.

Troubleshooting
- "gRPC stubs not found" — run `make gen` inside `edge/` and ensure `grpcio-tools` is installed.
- "failed to open capture source" — check your camera index and permissions (or pass a file path).
- If frames appear slow: reduce `--width/--height` and `--quality` on `sensor.py`.

Next steps you might want
1) Plug a real detector: add a small ONNX model call in `infer_server.py` and return real detections.
2) Switch proto to use raw bytes (remove base64) to lower bandwidth.
3) Add an automated integration test that spins up the server and a fake client and asserts a
   DetectionResponse is received.

Files
- `edge/infer_server.py`
- `edge/sensor.py`
- `edge/proto/infer.proto`
- `edge/generate_stubs.py`
- `edge/Makefile`

License / notes
- Minimal demo code provided as-is for local testing. Not hardened for production use.
