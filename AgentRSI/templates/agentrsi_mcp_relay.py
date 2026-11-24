#!/usr/bin/env python3
"""Stdlib-only stdio <-> AF_UNIX MCP JSON-RPC relay for AgentRSI containers."""
from __future__ import annotations

import argparse
import socket
import sys
import threading


def _pump_stdin_to_sock(sock: socket.socket) -> None:
    stdin = sys.stdin.buffer
    try:
        while True:
            line = stdin.readline()
            if not line:
                break
            sock.sendall(line)
    except OSError:
        pass
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def relay(socket_path: str) -> int:
    """Connect to `socket_path` and bidirectionally relay newline-delimited messages."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)
    except OSError as exc:
        print(f"agentrsi_mcp_relay: failed to connect to {socket_path}: {exc}", file=sys.stderr)
        try:
            sock.close()
        except OSError:
            pass
        return 1

    writer_thread = threading.Thread(
        target=_pump_stdin_to_sock,
        args=(sock,),
        name="mcp-relay-stdin",
        daemon=True,
    )
    writer_thread.start()

    stdout = sys.stdout.buffer
    try:
        with sock.makefile("rb") as sock_reader:
            while True:
                line = sock_reader.readline()
                if not line:
                    break
                stdout.write(line)
                stdout.flush()
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stdio-to-Unix-socket MCP JSON-RPC relay for AgentRSI."
    )
    ap.add_argument(
        "--socket",
        default="/run/agentrsi/colab-mcp.sock",
        help="Path to the mounted AF_UNIX MCP broker socket.",
    )
    args = ap.parse_args()
    raise SystemExit(relay(args.socket))


if __name__ == "__main__":
    main()
