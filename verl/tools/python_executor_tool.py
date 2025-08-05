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
    Python Code Execution Tool
    
    This tool connects to a Python code execution service via MCP protocol,
    allowing execution of Python code in a secure environment and returning results.
    """
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        print(f"Initializing PythonExecutorTool with config: {config}")

    def _parse_tool_result(self, content: list) -> tuple[str, dict]:
        """
        Parse tool execution result
        
        Supports two formats:
        1. Old format: JSON format {"success": bool, "result": str, "stdout": str, "stderr": str}
        2. New format: XML format <status>success</status><dependencies>[]</dependencies><o>output</o><return_value>value</return_value>
        
        Args:
            content: Content list returned by MCP service
            
        Returns:
            tuple: (formatted result string, metadata dict)
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
                
                # 检查是否为XML格式（新的Pydantic MCP格式）
                if "<status>" in text and "</status>" in text:
                    result_text, metadata = self._parse_xml_result(text, metadata)
                else:
                    # 尝试解析JSON格式（旧格式）
                    try:
                        result_data = json.loads(text)
                        
                        if isinstance(result_data, dict):
                            result_text, metadata = self._parse_json_result(result_data, metadata)
                        else:
                            result_text = str(result_data)
                            metadata["status"] = "success"
                            
                    except json.JSONDecodeError:
                        # 如果不是JSON格式，直接使用原始文本
                        result_text = text
                        metadata["status"] = "success"
                    
        except Exception as e:
            error_msg = f"Error parsing tool result: {str(e)}"
            print(error_msg)
            metadata["api_request_error"] = error_msg
            metadata["status"] = "error"
            result_text = f"❌ {error_msg}"

        # if random.randint(0, 100) < 1:
        print("================")
        print(metadata)
        print("================")
        print(result_text)
        print("================")
        
        return result_text, metadata
    
    def _parse_xml_result(self, text: str, metadata: dict) -> tuple[str, dict]:
        """
        解析XML格式的结果（Pydantic MCP格式）
        
        XML格式示例：
        <status>success</status>
        <dependencies>["numpy"]</dependencies>
        <o>output content</o>
        <return_value>return value</return_value>
        
        或错误格式：
        <status>run-error</status>
        <error>error message with traceback</error>
        """
        import re
        
        # 提取status
        status_match = re.search(r'<status>([^<]+)</status>', text)
        status = status_match.group(1) if status_match else "unknown"
        
        formatted_result = []
        
        if status == "success":
            formatted_result.append("✅ Code executed successfully")
            metadata["execution_success"] = True
            metadata["status"] = "success"
            
            # 提取dependencies
            deps_match = re.search(r'<dependencies>([^<]*)</dependencies>', text)
            if deps_match and deps_match.group(1).strip():
                formatted_result.append(f"📦 Dependencies: {deps_match.group(1)}")
            
            # 提取output
            output_match = re.search(r'<o>(.*?)</o>', text, re.DOTALL)
            if output_match and output_match.group(1).strip():
                output_content = output_match.group(1).strip()
                # 过滤 Loading/Loaded 消息
                output_content = self._filter_loading_messages(output_content)
                if output_content:  # 只有过滤后还有内容才显示
                    formatted_result.append(f"\n📤 Output:\n{output_content}")
                    metadata["has_output"] = True
            
            # 提取return_value
            return_match = re.search(r'<return_value>(.*?)</return_value>', text, re.DOTALL)
            if return_match and return_match.group(1).strip():
                return_content = return_match.group(1).strip()
                formatted_result.append(f"\n↩️ Return value:\n{return_content}")
            
            # 如果执行成功但没有输出和返回值，添加提示
            if not output_match and not return_match:
                formatted_result.append("\n(Code execution completed, no output)")
                
        else:
            # 处理错误情况
            if status in ["install-error", "run-error"]:
                formatted_result.append("❌ Code execution failed")
                
                # 提取错误信息
                error_match = re.search(r'<error>(.*?)</error>', text, re.DOTALL)
                if error_match:
                    error_content = error_match.group(1).strip()
                    # 过滤 Loading/Loaded 消息
                    error_content = self._filter_loading_messages(error_content)
                    formatted_result.append(f"Error: {error_content}")
                else:
                    formatted_result.append(f"Status: {status}")
                
                metadata["execution_success"] = False
                metadata["status"] = "error"
                metadata["has_error"] = True
            else:
                formatted_result.append(f"⚠️ Unknown status: {status}")
                metadata["status"] = "unknown"
        
        return "\n".join(formatted_result), metadata
    
    def _filter_loading_messages(self, text: str) -> str:
        """
        过滤掉 Pyodide 环境的 Loading/Loaded 消息
        示例：
        输入："Loading numpyLoaded numpyLoading mpmath, sympyLoaded mpmath, sympy55.0"
        输出："55.0"
        """
        import re
        
        # 使用 findall 找到所有 Loading/Loaded 对
        matches = re.findall(r'Loading ([a-zA-Z0-9_,\s]+?)Loaded \1', text)
        
        # 逐个删除每个匹配的 Loading/Loaded 对
        for package_list in matches:
            package_list = package_list.strip()
            # 删除 "Loading 包名列表"
            text = text.replace(f"Loading {package_list}", "")
            # 删除 "Loaded 包名列表" 
            text = text.replace(f"Loaded {package_list}", "")
        
        # 清理多余的空白字符
        text = text.strip()
        
        return text

    def _parse_json_result(self, result_data: dict, metadata: dict) -> tuple[str, dict]:
        """
        解析JSON格式的结果（旧格式）
        """
        success = result_data.get("success", False)
        result_msg = result_data.get("result", "")
        stdout = result_data.get("stdout", "")
        stderr = result_data.get("stderr", "")
        
        # 过滤 stdout 和 stderr 中的 Loading/Loaded 消息
        stdout = self._filter_loading_messages(stdout)
        # stderr = self._filter_loading_messages(stderr)
        
        # 构建格式化的结果
        formatted_result = []
        
        if success:
            formatted_result.append("✅ Code executed successfully")
            metadata["execution_success"] = True
            metadata["status"] = "success"
        else:
            formatted_result.append("❌ Code execution failed")
            formatted_result.append(f"Error: {result_msg}")
            metadata["execution_success"] = False
            metadata["status"] = "error"
            metadata["has_error"] = True
        
        # 添加标准输出
        if stdout.strip():
            formatted_result.append(f"\n📤 Output:\n{stdout}")
            metadata["has_output"] = True
        
        # 添加错误输出
        if stderr.strip():
            formatted_result.append(f"\n⚠️ Error output:\n{stderr}")
            metadata["has_error"] = True
        
        # 如果执行成功但没有输出，添加提示
        if success and not stdout.strip() and not stderr.strip():
            formatted_result.append("\n(Code execution completed, no output)")
        
        return "\n".join(formatted_result), metadata