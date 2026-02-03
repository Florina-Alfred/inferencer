#!/usr/bin/env python3
"""Minimal client that connects to the inference server and sends no frames.

This file is a small placeholder. Use it to verify connectivity once stubs
are generated. It intentionally avoids heavy runtime logic.
"""
import argparse
import time

import grpc

try:
    from edge.proto import infer_pb2, infer_pb2_grpc
except Exception as exc:
    raise RuntimeError("gRPC stubs missing: run `python generate_stubs.py` in edge/") from exc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=50051, type=int)
    args = p.parse_args()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = infer_pb2_grpc.InferenceStub(channel)

    # simple test: open a stream and close it immediately
    def gen():
        # send one empty ImageRequest then stop
        req = infer_pb2.ImageRequest(source="test", image_b64="", width=0, height=0, timestamp_ms=int(time.time()*1000), seq=0)
        yield req

    responses = stub.Stream(gen())
    for resp in responses:
        print("got response", resp)


if __name__ == "__main__":
    main()
