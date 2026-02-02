import sys

from inference_worker import run_inference_stream


def main(argv: list[str]) -> None:
    source = argv[1] if len(argv) > 1 else "0"
    client_addr = argv[2] if len(argv) > 2 else "127.0.0.1"
    client_port = int(argv[3]) if len(argv) > 3 else 56060
    model = argv[4] if len(argv) > 4 else "l"
    run_inference_stream(source, client_addr, client_port, model)


if __name__ == "__main__":
    main(sys.argv)
