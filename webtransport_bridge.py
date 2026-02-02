"""
webtransport_bridge.py

Lightweight WebTransport <-> UDP bridge for local testing.

Features:
- Acts as a UDP client to the existing `udp_infer_server.py` by sending
  `START|<source>|<port>` / `HEARTBEAT|<source>|<port>` / `STOP|<source>|<port>`
  for each subscribed `source`.
- Exposes a WebTransport server (aioquic) that accepts browser and Python
  WebTransport clients. Clients subscribe by connecting to
  `https://host:port/?source=<n>` (query param) where `<n>` is the source id.
- For each subscribed source the bridge binds a dedicated UDP socket so
  incoming datagrams can be mapped back to the source and forwarded to
  connected WebTransport clients.

Notes / caveats:
- This is a development/testing bridge; QUIC/TLS details are simplified.
- It auto-generates a self-signed cert/key (cert.pem/key.pem) when none
  are found and `openssl` is available.
- The aioquic WebTransport API surface used here is intentionally small and
  the implementation is a pragmatic, best-effort bridge for local testing.

Usage:
  pip install aioquic
  python webtransport_bridge.py --host 0.0.0.0 --port 4433 --udp-server 127.0.0.1:55055

Browser test (Chromium-based):
  - Start bridge (above). It will generate cert.pem/key.pem in cwd if missing.
  - Open Chrome/Chromium with insecure-localhost allowed (or import cert):
    chrome --user-data-dir=/tmp/ct --allow-insecure-localhost
  - Open the demo `webtransport_demo.html` served from any local file server
    and point it at the bridge `https://localhost:4433/?source=0`.

"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import threading
import time
from typing import Dict, Set, Tuple

from aioquic.asyncio.server import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h3.connection import H3Connection
from aioquic.h3.events import HeadersReceived, DatagramReceived

CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"


def generate_self_signed_cert(cert: str = CERT_FILE, key: str = KEY_FILE) -> None:
    """Generate a self-signed cert/key pair using openssl if available."""
    if os.path.exists(cert) and os.path.exists(key):
        return
    print("Generating self-signed TLS cert/key (cert.pem/key.pem)...")
    try:
        subprocess.check_call(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                key,
                "-out",
                cert,
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ]
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        print("Failed to generate cert automatically:", exc)
        print("Please create cert.pem/key.pem manually and re-run.")
        raise


class WTBridgeProtocol(QuicConnectionProtocol):
    """QUIC protocol that integrates a minimal H3 handler and exposes
    a WebTransportSession when the client connects with WebTransport.

    This class accepts a WebTransport connection where the client passes
    the desired `source` as a query parameter in the :path header, e.g.
    `/?source=0`.
    """

    def __init__(self, *args, bridge: "WebTransportBridge", **kwargs):
        # enable WebTransport support in H3Connection
        super().__init__(*args, **kwargs)
        self._http = H3Connection(self._quic, enable_webtransport=True)
        self._bridge = bridge
        # track active stream_ids that belong to this connection
        self._streams: set[int] = set()

    def quic_event_received(self, event):
        # Let H3Connection handle events and surface H3/WT events via
        # `handle_event`. The H3 events we care about are HeadersReceived
        # which contain the initial request (including the path/query).
        for http_event in self._http.handle_event(event):
            # Register new request streams that contain the source query param
            if isinstance(http_event, HeadersReceived):
                headers = {k.decode(): v.decode() for k, v in http_event.headers}
                path = headers.get(":path", "")
                source = None
                if "source=" in path:
                    try:
                        source = int(path.split("source=")[1].split("&")[0])
                    except Exception:
                        source = None
                if source is None:
                    continue
                # store stream id and register with bridge; bridge will use
                # the H3Connection instance and stream id to send datagrams.
                stream_id = http_event.stream_id
                self._streams.add(stream_id)
                self._bridge.register_session(source, self, stream_id)

            # Optionally handle datagrams from client (not used here)
            if isinstance(http_event, DatagramReceived):
                # client datagram associated with some stream; ignore for now
                pass


class WebTransportBridge:
    """High level bridge manager.

    Maintains per-source UDP sockets and WebTransport session subscribers.
    """

    def __init__(self, udp_server_addr: Tuple[str, int], host: str = "0.0.0.0", port: int = 4433):
        self.udp_server_addr = udp_server_addr
        self._host = host
        self._port = port

        # source -> (socket, thread)
        self._sockets: Dict[int, socket.socket] = {}
        self._recv_threads: Dict[int, threading.Thread] = {}

        # source -> set of (protocol, stream_id)
        self._sessions: Dict[int, Set[Tuple["WTBridgeProtocol", int]]] = {}

        # heartbeat: source -> last_heartbeat_time
        self._last_heartbeat: Dict[int, float] = {}

        self._lock = threading.Lock()

    def register_session(self, source: int, protocol: "WTBridgeProtocol", stream_id: int) -> None:
        """Register a new (protocol, stream_id) tuple for a source.

        The protocol holds the H3Connection instance with `send_datagram`.
        """
        print(f"Registering WT session for source={source} stream={stream_id}")
        with self._lock:
            sessions = self._sessions.setdefault(source, set())
            sessions.add((protocol, stream_id))
            if source not in self._sockets:
                # create a UDP socket for this source and register with server
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(("0.0.0.0", 0))
                port = sock.getsockname()[1]
                self._sockets[source] = sock
                # send START for this source to udp server
                self._send_udp_control("START", source, port)
                # start receiver thread
                t = threading.Thread(target=self._recv_loop, args=(source, sock), daemon=True)
                self._recv_threads[source] = t
                t.start()

    def unregister_session(self, source: int, protocol: "WTBridgeProtocol", stream_id: int) -> None:
        with self._lock:
            sessions = self._sessions.get(source)
            if not sessions:
                return
            sessions.discard((protocol, stream_id))
            if not sessions:
                # last client left -> send STOP and close socket
                self._send_udp_control("STOP", source, self._sockets[source].getsockname()[1])
                try:
                    self._sockets[source].close()
                except Exception:
                    pass
                del self._sockets[source]
                # thread will exit due to socket close

    def _send_udp_control(self, cmd: str, source: int, port: int) -> None:
        msg = f"{cmd}|{source}|{port}".encode()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(msg, self.udp_server_addr)
        print(f"Sent {cmd} for source={source} port={port} to {self.udp_server_addr}")

    def _recv_loop(self, source: int, sock: socket.socket) -> None:
        print(f"Started UDP recv loop for source={source} on {sock.getsockname()}")
        try:
            while True:
                data, addr = sock.recvfrom(65535)
                # forward to sessions subscribed to this source
                with self._lock:
                    sessions = list(self._sessions.get(source, []))
                for protocol, stream_id in sessions:
                    try:
                        # use H3Connection.send_datagram and flush via protocol.transmit()
                        protocol._http.send_datagram(stream_id, data)
                        protocol.transmit()
                    except Exception:
                        # some session/connection may be closed; ignore here
                        pass
        except Exception:
            # socket likely closed
            pass

    async def run(self) -> None:
        config = QuicConfiguration(is_client=False, alpn_protocols=["h3-29"])  # h3 draft
        config.load_cert_chain(CERT_FILE, KEY_FILE)

        print(f"Starting WebTransport bridge on https://{self._host}:{self._port}")
        await serve(
            self._host,
            self._port,
            configuration=config,
            create_protocol=lambda *args, **kwargs: WTBridgeProtocol(*args, bridge=self, **kwargs),
        )


def periodic_heartbeat(bridge: WebTransportBridge, interval: float = 5.0) -> None:
    """Periodically send HEARTBEAT for each active source to keep server happy."""
    while True:
        with bridge._lock:
            sources = list(bridge._sockets.keys())
        for source in sources:
            try:
                port = bridge._sockets[source].getsockname()[1]
                bridge._send_udp_control("HEARTBEAT", source, port)
            except Exception:
                pass
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4433)
    parser.add_argument("--udp-server", required=True, help="UDP server host:port (e.g. 127.0.0.1:55055)")
    args = parser.parse_args(argv)

    udp_host, udp_port = args.udp_server.split(":")
    udp_port = int(udp_port)

    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        try:
            generate_self_signed_cert()
        except Exception:
            print("Continuing without certs will fail; please provide cert.pem/key.pem")
            return 2

    bridge = WebTransportBridge((udp_host, udp_port), host=args.host, port=args.port)

    hb = threading.Thread(target=periodic_heartbeat, args=(bridge,), daemon=True)
    hb.start()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bridge.run())
    except KeyboardInterrupt:
        print("Shutting down bridge")
    finally:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
