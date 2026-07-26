"""
This module contains the MCPClient class, which is used to interact with the MCP server.
"""
import os
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    """
    A class that represents the MCP client.
    """
    def __init__(self, server_script_paths: list[dict] = None):
        self.server_script_paths = server_script_paths
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.session_manager = dict()

    async def connect_to_server(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        for mcp_server in self.server_script_paths:
            server_name = mcp_server["name"]
            server_script_path = mcp_server["server_script_path"]

            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env={
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
                }
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session_manager[server_name] = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session_manager[server_name].initialize()

            # List available tools
            response = await self.session_manager[server_name].list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def get_available_tools(self):
        """Get available tools from the MCP server
        """
        available_tools = []
        for server_name, session in self.session_manager.items():
            response = await session.list_tools()
            for tool in response.tools:
                available_tools.append({
                    "server_name": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        return available_tools

    async def tool_call(self, server_name: str, tool_name: str, tool_args: dict):
        """Call a tool
        """
        response = await self.session_manager[server_name].call_tool(tool_name, tool_args)
        return response.content[0].text

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
