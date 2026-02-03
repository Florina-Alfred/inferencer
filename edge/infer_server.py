#!/usr/bin/env python3
"""Minimal gRPC inference server (placeholder).

This server is intentionally small: it expects the generated stubs to be
present in `edge/proto`. If the imports fail, run `python generate_stubs.py`.
"""
import argparse
import time
from concurrent import futures

import grpc


try:
    from edge.proto import infer_pb2, infer_pb2_grpc
except Exception as exc:  # pragma: no cover - helpful error
    raise RuntimeError("gRPC stubs missing: run `python generate_stubs.py` in edge/") from exc


class InferenceServicer(infer_pb2_grpc.InferenceServicer):
    def Stream(self, request_iterator, context):
        for req in request_iterator:
            # echo back an empty response with timestamps; real inference goes here
            resp = infer_pb2.DetectionResponse(
                source=req.source,
                timestamp_ms=int(time.time() * 1000),
                seq=getattr(req, "seq", 0),
                processing_ms=0,
            )
            yield resp


def serve(host: str, port: int, workers: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    infer_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    server.start()
    print("server started on", bind)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", default=50051, type=int)
    p.add_argument("--workers", default=4, type=int)
    args = p.parse_args()
    serve(args.host, args.port, args.workers)


if __name__ == "__main__":
    main()
