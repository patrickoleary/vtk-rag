"""Persistent MCP client that maintains a single stdio session across multiple tool calls."""

from __future__ import annotations

import asyncio
import atexit
import threading
from concurrent.futures import Future
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult


class PersistentMCPClient:
    """Thin wrapper that keeps a single MCP stdio session alive across calls."""

    def __init__(self, server: StdioServerParameters) -> None:
        """Initialize the client and start the MCP session in a background thread.

        Args:
            server: MCP server parameters for stdio communication.
        """
        self._server = server
        # Create a dedicated asyncio event loop for the MCP session.
        self._loop = asyncio.new_event_loop()
        # Threading events to coordinate startup and shutdown.
        self._ready = threading.Event()
        self._closed = threading.Event()
        # Spin up a background thread to run the event loop.
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Block until the MCP session is initialized and ready.
        self._ready.wait()
        # Ensure we clean up the session when the process exits.
        atexit.register(self.close)

    def _run_loop(self) -> None:
        """Entry point for the background thread: set up the loop and run the main coroutine."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._loop_main())
        self._loop.close()

    async def _loop_main(self) -> None:
        """Main coroutine: start the MCP stdio session, process requests from the queue, and clean up."""
        # Create a queue to receive (tool_name, payload, future) tuples from other threads.
        self._queue = asyncio.Queue()
        # Launch the MCP server process and establish stdio communication.
        self._stdio_cm = stdio_client(self._server)
        read, write = await self._stdio_cm.__aenter__()
        try:
            # Open an MCP client session over the stdio streams.
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            # Initialize the session (handshake with the server).
            await self._session.initialize()
            # Signal the main thread that the session is ready.
            self._ready.set()
            # Process requests until we receive a shutdown sentinel (name=None).
            while True:
                name, payload, future = await self._queue.get()
                if name is None:
                    # Shutdown sentinel: acknowledge and break out of the loop.
                    if future:
                        future.set_result(None)
                    break
                try:
                    # Call the MCP tool and return the result via the future.
                    result = await self._session.call_tool(name, payload)
                except Exception as exc:  # pragma: no cover - passthrough
                    future.set_exception(exc)
                else:
                    future.set_result(result)
        finally:
            # Clean up the session and stdio connection (must happen in the same task).
            if hasattr(self, "_session_cm"):
                await self._session_cm.__aexit__(None, None, None)
            await self._stdio_cm.__aexit__(None, None, None)
            self._closed.set()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call an MCP tool and return the result (blocks until complete)."""
        if self._closed.is_set():
            raise RuntimeError("MCP client is closed")
        # Create a future to receive the result.
        result_future = Future()
        # Enqueue the tool call request.
        asyncio.run_coroutine_threadsafe(
            self._queue.put((name, arguments, result_future)), self._loop
        ).result()
        # Block until the result is ready and return it.
        return result_future.result()

    def close(self) -> None:
        """Shut down the MCP session and join the background thread."""
        if self._closed.is_set():
            return
        # Enqueue a shutdown sentinel (name=None) to tell the worker loop to exit.
        sentinel_done = Future()
        asyncio.run_coroutine_threadsafe(
            self._queue.put((None, None, sentinel_done)), self._loop
        ).result()
        # Wait for the worker to acknowledge shutdown.
        sentinel_done.result()
        # Wait for the background thread to finish.
        self._thread.join()
        self._closed.set()
