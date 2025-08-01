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
import io
import logging
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# 创建 MCP 服务器
mcp = FastMCP("Python Code Executor")


@mcp.tool()
def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    执行 Python 代码并返回结果
    
    Args:
        code: 要执行的 Python 代码
        timeout: 执行超时时间（秒），默认30秒
        
    Returns:
        Dict containing:
        - success: 是否执行成功
        - result: 执行结果或错误信息
        - stdout: 标准输出
        - stderr: 标准错误输出
    """
    # 安全检查 - 禁止某些危险操作
    forbidden_imports = ['os', 'subprocess', 'sys', 'importlib', '__import__']
    forbidden_functions = ['exec', 'eval', 'open', 'file', 'input', 'raw_input']
    
    # 检查代码中是否包含危险的导入或函数
    code_lines = code.lower()
    for forbidden in forbidden_imports + forbidden_functions:
        if forbidden in code_lines:
            return {
                "success": False,
                "result": f"Error: 禁止使用 '{forbidden}' 以确保安全",
                "stdout": "",
                "stderr": ""
            }
    
    # 创建输出捕获器
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # 创建安全的执行环境
        safe_globals = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'print': print,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'dir': dir,
                'vars': vars,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
            },
            # 允许一些常用的数学和数据处理库
            'math': __import__('math'),
            'json': __import__('json'),
            'datetime': __import__('datetime'),
            'random': __import__('random'),
        }
        
        # 尝试导入numpy和pandas（如果可用）
        try:
            safe_globals['numpy'] = __import__('numpy')
            safe_globals['np'] = safe_globals['numpy']
        except ImportError:
            pass
            
        try:
            safe_globals['pandas'] = __import__('pandas')
            safe_globals['pd'] = safe_globals['pandas']
        except ImportError:
            pass
        
        # 执行代码并捕获输出
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # 使用 compile 和 exec 执行代码
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, safe_globals)
        
        return {
            "success": True,
            "result": "代码执行成功",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue()
        }
        
    except SyntaxError as e:
        return {
            "success": False,
            "result": f"语法错误: {str(e)}",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue()
        }
    except Exception as e:
        return {
            "success": False,
            "result": f"执行错误: {str(e)}\n{traceback.format_exc()}",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue()
        }


@mcp.tool()
def get_available_modules() -> Dict[str, Any]:
    """
    获取可用的模块列表
    
    Returns:
        Dict containing available modules and their descriptions
    """
    available_modules = {
        "built_in": {
            "description": "内置函数和类型",
            "items": ["len", "str", "int", "float", "bool", "list", "dict", "tuple", "set", 
                     "range", "enumerate", "zip", "map", "filter", "sum", "max", "min", 
                     "abs", "round", "print", "type", "isinstance", "hasattr", "getattr", 
                     "setattr", "dir", "vars", "sorted", "reversed", "any", "all"]
        },
        "math": {
            "description": "数学函数和常量",
            "items": ["sin", "cos", "tan", "sqrt", "log", "exp", "pi", "e"]
        },
        "json": {
            "description": "JSON 编码和解码",
            "items": ["loads", "dumps"]
        },
        "datetime": {
            "description": "日期和时间处理",
            "items": ["datetime", "date", "time", "timedelta"]
        },
        "random": {
            "description": "随机数生成",
            "items": ["random", "randint", "choice", "shuffle"]
        }
    }
    
    # 检查可选模块
    try:
        import numpy
        available_modules["numpy"] = {
            "description": "数值计算库",
            "items": ["array", "zeros", "ones", "arange", "linspace", "mean", "std", "sum"]
        }
    except ImportError:
        available_modules["numpy"] = {
            "description": "未安装",
            "items": []
        }
    
    try:
        import pandas
        available_modules["pandas"] = {
            "description": "数据分析库",
            "items": ["DataFrame", "Series", "read_csv", "read_json"]
        }
    except ImportError:
        available_modules["pandas"] = {
            "description": "未安装", 
            "items": []
        }
    
    return {
        "success": True,
        "available_modules": available_modules
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Code Executor MCP Server")
    parser.add_argument("--port", type=int, help="Server port (if not specified, use STDIO)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    args = parser.parse_args()
    
    # 根据参数决定启动模式
    if args.port:
        # HTTP 模式
        print(f"Starting Python Code Executor MCP Server on {args.host}:{args.port}")
        mcp.run(host=args.host, port=args.port)
    else:
        # STDIO 模式（适合进程间通信）
        print("Starting Python Code Executor MCP Server in STDIO mode", file=sys.stderr)
        mcp.run()