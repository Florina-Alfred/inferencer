"""A minimal aioquic-based WebTransport test client.

This script connects to the bridge at the given URL and reads H3 datagrams
that the bridge forwards from the UDP server. It demonstrates the aioquic
client usage for receiving datagrams mapped to a request stream.

Usage:
  python test_webtransport_client.py --host localhost --port 4433 --source 0 --cert cert.pem

Note: This client accepts the provided `cert.pem` as the remote CA for
development testing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from aioquic.asyncio.client import connect
from aioquic.quic.configuration import QuicConfiguration


async def run(host: str, port: int, source: int, cert: str | None):
    conf = QuicConfiguration(is_client=True, alpn_protocols=["h3-29"])
    if cert:
        conf.load_verify_locations(cert)
    else:
        conf.verify_mode = False

    async with connect(host, port, configuration=conf, create_protocol=None):
        print("Connected; waiting for events")
        # Note: a full WebTransport client would perform an HTTP/3 request to
        # negotiate a WebTransport session. For this simple test we only
        # establish the QUIC/H3 connection and keep it open to observe that the
        # bridge can accept connections. Implementing a full client requires
        # sending the proper CONNECT/HEADERS frames and handling session ids.
        await asyncio.sleep(2)
        print("Test client finished (no datagram loop implemented).")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=4433)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--cert")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.host, args.port, args.source, args.cert))


if __name__ == "__main__":
    main()
