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

import json
import logging
import os
import re
import random

from verl.tools.mcp_base_tool import MCPBaseTool

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PythonExecutorTool(MCPBaseTool):
    """
    Python 代码执行工具
    
    该工具通过 MCP 协议连接到 Python 代码执行服务，
    允许在安全的环境中执行 Python 代码并返回结果。
    """
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        logger.info(f"初始化 PythonExecutorTool，配置: {config}")

    def _parse_tool_result(self, content: list) -> tuple[str, dict]:
        """
        解析工具执行结果
        
        Args:
            content: MCP 服务返回的内容列表
            
        Returns:
            tuple: (格式化的结果字符串, 元数据字典)
        """
        result_text = ""
        metadata = {
            "api_request_error": "",
            "status": "unknown",
            "execution_success": False,
            "has_output": False,
            "has_error": False,
        }
        
        try:
            for part in content:
                if part.type != "text":
                    continue
                    
                text = part.text
                
                # 尝试解析JSON响应
                try:
                    result_data = json.loads(text)
                    
                    if isinstance(result_data, dict):
                        success = result_data.get("success", False)
                        result_msg = result_data.get("result", "")
                        stdout = result_data.get("stdout", "")
                        stderr = result_data.get("stderr", "")
                        
                        # 构建格式化的结果
                        formatted_result = []
                        
                        if success:
                            formatted_result.append("✅ 代码执行成功")
                            metadata["execution_success"] = True
                            metadata["status"] = "success"
                        else:
                            formatted_result.append("❌ 代码执行失败")
                            formatted_result.append(f"错误: {result_msg}")
                            metadata["execution_success"] = False
                            metadata["status"] = "error"
                            metadata["has_error"] = True
                        
                        # 添加标准输出
                        if stdout.strip():
                            formatted_result.append(f"\n📤 输出:\n{stdout}")
                            metadata["has_output"] = True
                        
                        # 添加错误输出
                        if stderr.strip():
                            formatted_result.append(f"\n⚠️ 错误输出:\n{stderr}")
                            metadata["has_error"] = True
                        
                        # 如果执行成功但没有输出，添加提示
                        if success and not stdout.strip() and not stderr.strip():
                            formatted_result.append("\n(代码执行完成，无输出)")
                        
                        result_text = "\n".join(formatted_result)
                    else:
                        result_text = str(result_data)
                        metadata["status"] = "success"
                        
                except json.JSONDecodeError:
                    # 如果不是JSON格式，直接使用原始文本
                    result_text = text
                    metadata["status"] = "success"
                    
        except Exception as e:
            error_msg = f"解析工具结果时出错: {str(e)}"
            logger.error(error_msg)
            metadata["api_request_error"] = error_msg
            metadata["status"] = "error"
            result_text = f"❌ {error_msg}"

        if random.randint(0, 100) < 1:
            print("================")
            print(metadata)
            print("================")
            print(result_text)
            print("================")
        
        return result_text, metadata