#!/usr/bin/env python3
"""
测试 standalone_python_executor_server.py 的 HTTP 模式
演示如何正确使用 FastMCP 2.x 的 Streamable HTTP 协议
"""

import asyncio
import json
from fastmcp import Client

def _filter_loading_messages(text: str) -> str:
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

def _parse_json_result(result_data: dict, metadata: dict) -> tuple[str, dict]:
    """
    解析JSON格式的结果（旧格式）
    """
    success = result_data.get("success", False)
    result_msg = result_data.get("result", "")
    stdout = result_data.get("stdout", "")
    stderr = result_data.get("stderr", "")

    print(f"stdout: {stdout}")
    
    # 过滤 stdout 和 stderr 中的 Loading/Loaded 消息
    stdout = _filter_loading_messages(stdout)
    print(f"Processed stdout: {stdout}")
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

async def test_http_mode():
    """测试 HTTP 模式的 MCP 服务器"""
    print("🚀 测试 standalone_python_executor_server.py HTTP 模式")
    print("=" * 60)
    
    # 连接到 HTTP 服务器
    url = "http://127.0.0.1:3001/mcp"
    print(f"📡 连接到: {url}")
    
    try:
        # 使用 FastMCP 2.x 的 Streamable HTTP 客户端
        client = Client(url)
        
        async with client:
            print("✅ 成功连接到服务器")
            
            # 获取可用工具
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            print(f"📝 可用工具: {tool_names}")
            
            # 测试执行 Python 代码
            print("\n📝 测试执行 Python 代码:")
            test_code = """
import math
import json
import sympy
import numpy as np

# 计算一些数学函数
result = {
    'sqrt_16': math.sqrt(16),
    'pi': math.pi,
    'sin_pi_2': math.sin(math.pi/2)
}

print(f"计算结果: {json.dumps(result, indent=2)}")
"""
            
            result = await client.call_tool("execute_python", {
                "code": test_code.strip()
            })
            
            result_text = ""
            metadata = {
                "api_request_error": "",
                "status": "unknown",
                "execution_success": False,
                "has_output": False,
                "has_error": False,
            }
            # 显示执行结果
            if hasattr(result, 'content') and result.content:
                output = result.content[0].text
                result_data = json.loads(output)
                print(f"执行成功: {output}")
                result_text, metadata = _parse_json_result(result_data, metadata)
                print(result_text)
            
            print("\n🎉 HTTP 模式测试成功！")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n🔧 请确保:")
        print("1. 服务器已启动: python standalone_python_executor_server.py --host 127.0.0.1 --port 8002")
        print("2. 端口 8002 没有被占用")
        print("3. FastMCP 版本 >= 2.0")


if __name__ == "__main__":
    asyncio.run(test_http_mode())