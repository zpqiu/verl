# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import random
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from verl.tools.utils.mcp_clients.utils import TokenBucket, mcp2openai

logger = logging.getLogger(__name__)


class MCPClientManager:
    rootServerName = "mcpServers"
    
    def __init__(self):
        # 进程级别的初始化状态
        self.process_id = os.getpid()
        self.initialized = False
        self.clients = []
        self.tool_client_mapping = {}
        self.rate_limiter = None
        self.init_lock = asyncio.Lock()

    async def initialize(self, config_path, rate_limit: float = 10.0, max_retries: int = 3):
        async with self.init_lock:
            if self.initialized:
                print(f"[DEBUG][MCPClientManager] Process {self.process_id} already initialized, skipping")
                return
            
            """Initialize the MCP Client Manager and start all clients with retry logic"""
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} initializing with config: {config_path}")
            
            # 添加随机延迟以避免并发连接冲突
            delay = random.uniform(0.1, 2.0)
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} waiting {delay:.2f}s before initialization")
            await asyncio.sleep(delay)
            
            for attempt in range(max_retries):
                try:
                    await self._do_initialize(config_path, rate_limit)
                    print(f"[DEBUG][MCPClientManager] Process {self.process_id} successfully initialized on attempt {attempt + 1}")
                    break
                except Exception as e:
                    print(f"[DEBUG][MCPClientManager] Process {self.process_id} initialization attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        print(f"[ERROR][MCPClientManager] Process {self.process_id} failed to initialize after {max_retries} attempts")
                        raise
                    # 指数退避重试
                    retry_delay = random.uniform(1.0, 3.0) * (2 ** attempt)
                    print(f"[DEBUG][MCPClientManager] Process {self.process_id} retrying in {retry_delay:.2f}s")
                    await asyncio.sleep(retry_delay)
    
    async def _do_initialize(self, config_path, rate_limit: float):
        """实际的初始化逻辑"""
        result = self._load_config(config_path)
        servers = result[self.rootServerName]
        print(f"[DEBUG][MCPClientManager] Process {self.process_id} found servers: {servers}")
        
        exclude_sse_servers = {self.rootServerName: {}}
        for server_name in servers.keys():
            server = servers[server_name]
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} processing server {server_name}: {server}")
            if "auth_token" in server:
                transport = SSETransport(url=server["url"], headers={"Authorization": f"Bearer {server['auth_token']}"})
                client = Client(transport)
                self.clients.append(client)
                print(f"[DEBUG][MCPClientManager] Process {self.process_id} created SSE client for {server_name}")
            else:
                exclude_sse_servers[self.rootServerName][server_name] = server
                print(f"[DEBUG][MCPClientManager] Process {self.process_id} added {server_name} to STDIO servers")

        if exclude_sse_servers[self.rootServerName]:
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} creating STDIO client with config: {exclude_sse_servers}")
            client = Client(exclude_sse_servers)
            self.clients.append(client)
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} created STDIO client")

        print(f"[DEBUG][MCPClientManager] Process {self.process_id} total clients created: {len(self.clients)}")
        # Initialize rate limiter
        self.rate_limiter = TokenBucket(rate_limit)
        self.initialized = True

    async def call_tool(self, tool_name, parameters, timeout):
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        # print(f"[DEBUG][MCPClientManager] Process {self.process_id} call_tool: {tool_name}, {parameters}, {timeout}")
        client = self.get_client_with_tool_name(tool_name)
        # print(f"[DEBUG][MCPClientManager] client: {client}")
        # print(f"[DEBUG][MCPClientManager] client type: {type(client)}")
        try:
            async with client:
                # print(f"[DEBUG][MCPClientManager] Inside async context, about to call tool")
                result = await client.call_tool_mcp(tool_name, parameters, timeout=timeout)
                # print(f"[DEBUG][MCPClientManager] Tool call result: {result}")
                return result
        except Exception as e:
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} tool call failed with exception: {e}")
            import traceback
            print(f"[DEBUG][MCPClientManager] Process {self.process_id} traceback: {traceback.format_exc()}")
            raise

    async def fetch_tool_schemas(self, tool_selected_list: list[str]) -> list[dict]:
        tool_schemas = []
        for client in self.clients:
            async with client:
                tools = await client.list_tools_mcp()
                for tool in tools.tools:
                    if not tool_selected_list:
                        self.tool_client_mapping[tool.name] = client
                        tool_schemas.append(mcp2openai(tool))
                    elif tool.name in tool_selected_list:
                        self.tool_client_mapping[tool.name] = client
                        tool_schemas.append(mcp2openai(tool))

        return tool_schemas

    def get_client_with_tool_name(self, tool_name: str):
        return self.tool_client_mapping[tool_name]

    def _load_config(self, file: str) -> dict[str, Any]:
        try:
            with open(file) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f'the "{file}" file was not found')
        except Exception:
            print(f'there was an error reading the "{file}" file')

        return {}


# 创建进程级别的单例实例
ClientManager = MCPClientManager()
