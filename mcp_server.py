from __future__ import annotations

import os

from src.knowledge_mcp_server import build_server


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower() or "stdio"
    port = int(os.getenv("MCP_PORT", "8765"))
    host = os.getenv("MCP_HOST", "0.0.0.0" if transport == "sse" else "127.0.0.1").strip()

    server = build_server(host=host, port=port)
    if transport == "sse":
        server.run(transport="sse")
        return

    server.run()


if __name__ == "__main__":
    main()
