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
import ast
import logging
import sys
from typing import Any, Dict

from fastmcp import FastMCP

# 尝试导入 langchain-sandbox
try:
    from langchain_sandbox_zp import PyodideSandbox
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    print("[WARNING] langchain-sandbox 未安装，请运行: pip install langchain-sandbox", file=sys.stderr)

logger = logging.getLogger(__name__)

# 创建 MCP 服务器
mcp = FastMCP("Simple Python Sandbox")

# 全局沙盒实例
sandbox = None


async def get_sandbox():
    """获取或创建沙盒实例"""
    global sandbox
    if sandbox is None:
        if not SANDBOX_AVAILABLE:
            raise RuntimeError("langchain-sandbox 未安装")
        sandbox = PyodideSandbox(allow_net=True, stateful=False)
    return sandbox



def _prepare_code_for_execution(code: str) -> str:
    """
    准备代码执行，自动添加打印最后一个表达式的逻辑
    """
    try:
        # 解析AST
        tree = ast.parse(code)
        
        # 如果代码为空，直接返回
        if not tree.body:
            return code
            
        # 检查最后一个节点
        last_node = tree.body[-1]
        
        # 如果最后一个节点是表达式语句（不是赋值、导入等）
        if isinstance(last_node, ast.Expr):
            # 获取表达式内容
            expr = last_node.value
            
            # 检查是否为需要打印结果的表达式类型
            printable_expr_types = [ast.Call, ast.Name, ast.Attribute, ast.Subscript, 
                                   ast.BinOp, ast.Compare, ast.UnaryOp, ast.BoolOp,
                                   ast.List, ast.Dict, ast.Tuple, ast.Set]
            
            # 兼容不同Python版本的常量类型
            try:
                printable_expr_types.extend([ast.Constant, ast.Num, ast.Str])
            except AttributeError:
                # 旧版本Python可能没有ast.Constant
                pass
                
            if isinstance(expr, tuple(printable_expr_types)):
                
                # 获取原始代码的最后一行
                lines = code.strip().split('\n')
                last_line = lines[-1].strip()
                
                # 检查是否已经有print语句
                if not last_line.startswith('print('):
                    # 修改最后一行，将表达式赋值给变量并打印
                    lines[-1] = f"_last_expr_result = {last_line}"
                    lines.append("if _last_expr_result is not None:")
                    lines.append("    print(_last_expr_result)")
                    
                    return '\n'.join(lines)
                
        return code
        
    except SyntaxError:
        # 如果解析失败，返回原代码
        return code
    except Exception:
        # 其他异常也返回原代码
        return code


@mcp.tool()
async def execute_python(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox
    
    CRITICAL CODE GENERATION RULES:
    1. CODE PURITY: Generate ONLY executable Python code. NO comments, explanations, or thinking process in the code parameter.
    2. THINKING SEPARATION: Do your thinking BEFORE calling this tool, not inside the code parameter.
    3. CLEAN OUTPUT: The code should be production-ready, without any meta-commentary.
    
    IMPORTANT CODING GUIDELINES:
    - Avoid using f-strings (f"...") as they frequently cause SyntaxError: unterminated f-string literal
    - Use .format() method or % formatting instead: "Hello {}".format(name) or "Hello %s" % name
    - Use print() statements with comma-separated values: print("Value:", variable)
    - For string concatenation, use + operator: "Hello " + str(variable)
    - Always ensure proper string escaping and quote matching
    - NO inline comments explaining your thought process
    - NO TODO comments or placeholder comments
    - Code should be self-contained and immediately executable
    
    EXAMPLES OF WHAT NOT TO DO:
    ❌ BAD: "# First, I need to calculate the sum\nresult = 1 + 2\n# Now I'll print the result\nprint(result)"
    ❌ BAD: "# This solves the problem by...\nimport math\nmath.sqrt(16)"
    
    EXAMPLES OF CORRECT USAGE:
    ✅ GOOD: "result = 1 + 2\nprint(result)"
    ✅ GOOD: "import math\nprint(math.sqrt(16))"
    
    Args:
        code: Pure Python code to execute (NO comments or explanations)
        
    Returns:
        Dict containing:
        - success: Whether execution was successful
        - result: Execution result
        - stdout: Standard output
        - stderr: Standard error output
    """
    print(f"[DEBUG] execute code: {code[:50]}...", file=sys.stderr)
    
    if not SANDBOX_AVAILABLE:
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "langchain-sandbox not installed, please run: pip install langchain-sandbox"
        }
    
    try:
        sb = await get_sandbox()
        code = _prepare_code_for_execution(code)
        result = await sb.execute(code, timeout_seconds=30, memory_limit_mb=1024)
        
        return {
            "success": result.status == "success",
            "result": result.result,
            "stdout": result.stdout or "",
            "stderr": result.stderr or ""
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": f"execution error: {str(e)}"
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Python Sandbox MCP Server")
    parser.add_argument("--port", type=int, help="Server port (if not specified, use STDIO)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    args = parser.parse_args()
    
    if not SANDBOX_AVAILABLE:
        print("错误: langchain-sandbox 未安装")
        print("请运行: pip install langchain-sandbox")
        sys.exit(1)
    
    mcp.run(
        transport="http",
        host=args.host,
        port=args.port,
        path="/mcp",
        log_level="debug"
    )