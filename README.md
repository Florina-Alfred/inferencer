Inferencer — UDP inference demo

What it is
- Local demo: captures video, runs inference, and streams JSON detection datagrams over UDP to clients.

Quick start (commands)
- Prepare the environment and install with `uv`
  ```bash
  # from the repo root
  uv sync       # creates .venv and installs pinned deps
  source .venv/bin/activate
  ```
- (Optional) Regenerate lockfile after changing dependencies:
  ```bash
  uv lock
  ```
- Helper commands (useful when testing RTSP sources)
  - Run an RTSP server container and forward ports:
    ```bash
    docker run -p 9191:9191 -p 9192:9192 -p 8080:8080 ghcr.io/florina-alfred/rtsper:latest
    ```
  - Stream a local V4L2 device to the RTSP server (replace `/dev/video0` as needed):
    ```bash
    ffmpeg \
      -f v4l2 -video_size 640x480 -framerate 30 -i /dev/video0 \
      -vcodec libx264 -preset ultrafast -tune zerolatency \
      -crf 18 -pix_fmt yuv420p \
      -f rtsp -rtsp_transport tcp rtsp://localhost:9191/topic1
    ```
  - Stream a local file looped into the RTSP server (useful for testing):
    ```bash
    ffmpeg \
      -re -stream_loop -1 -i data/detection.mp4 \
      -vcodec libx264 -preset ultrafast -tune zerolatency \
      -crf 18 -pix_fmt yuv420p \
      -f rtsp -rtsp_transport tcp rtsp://localhost:9191/topic1
    ```
  - Play the RTSP stream locally with ffplay:
    ```bash
    ffplay -rtsp_transport tcp rtsp://localhost:9192/topic1
    ```
- Start the UDP inference server (default listens on 0.0.0.0:55055)
  ```bash
  uv run udp_infer_server.py
  ```
- Start a client that requests a job and listens for results
  - Use camera index 0 (OpenCV):
    ```bash
    uv run test_udp_client.py 0
    ```
  - Or use a device path /dev/video0 (FFmpeg backend):
    ```bash
    uv run test_udp_client.py /dev/video0
    ```
  - Optional second arg: receive port for the client (default 56060)


- Run a single-process demo (no server/worker split)
  ```bash
  uv run main.py -- --device cpu --model l --source 0
  ```

- Run an inference worker directly (useful for debugging)
  ```bash
  uv run run_worker.py 0 127.0.0.1 56060
  ```

- Run the viewer (opens the source and overlays detections received from the server)
  ```bash
  # RTSP example
  uv run viewer.py rtsp://localhost:9192/topic1 127.0.0.1 55055 56070

  # Local device example
  uv run viewer.py /dev/video0 127.0.0.1 55055 56071
  ```

Notes & troubleshooting
- This code sends JSON detection datagrams over UDP; clients must bind a UDP port and send a `START|<source>|<port>` message to the server to receive results. See `test_udp_client.py` for the exact control protocol.
- If camera capture fails: check permissions (`sudo usermod -aG video $USER`), ensure no other process holds the device, or try using the numeric camera index (`0`).
- If OpenCV capture with FFmpeg is required, ensure your OpenCV build includes FFmpeg.

Key files
- `udp_infer_server.py` — UDP server that manages jobs and forwards detections
- `inference_worker.py` — worker that captures frames, runs ONNX, and sends JSON results
- `test_udp_client.py` — example client: sends START/HEARTBEAT/STOP and prints received JSON
- `main.py` — single-process demo
- `pyproject.toml`, `uv.lock` — dependency manifest and lockfile
