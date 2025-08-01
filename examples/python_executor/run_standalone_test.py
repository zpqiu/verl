#!/usr/bin/env python3
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

"""
独立的 MCP 服务器测试脚本

这个脚本直接测试 MCP 服务器的功能，
不依赖 verl 的工具系统。
"""

import asyncio
import json
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


async def test_mcp_server_directly():
    """直接测试 MCP 服务器功能"""
    print("🔍 直接测试 MCP 服务器功能...")
    
    try:
        # 导入并运行 MCP 服务器功能
        from verl.tools.mcp_services.python_executor_server import execute_python_code, get_available_modules
        
        print("✅ 成功导入 MCP 服务器函数")
        
        # 测试代码执行
        print("\n📝 测试代码执行功能:")
        test_code = '''
import math
result = math.sqrt(16)
print(f"sqrt(16) = {result}")
for i in range(3):
    print(f"循环 {i}")
        '''.strip()
        
        print(f"执行代码:\n{test_code}")
        
        result = execute_python_code(test_code)
        print(f"执行结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 测试获取可用模块
        print("\n📝 测试获取可用模块:")
        modules_result = get_available_modules()
        print("可用模块:")
        print(json.dumps(modules_result, indent=2, ensure_ascii=False))
        
        # 测试错误处理
        print("\n📝 测试错误处理:")
        error_code = "result = 1 / 0"
        error_result = execute_python_code(error_code)
        print(f"错误处理结果:")
        print(json.dumps(error_result, indent=2, ensure_ascii=False))
        
        # 测试安全限制
        print("\n📝 测试安全限制:")
        unsafe_code = "import os; os.system('ls')"
        unsafe_result = execute_python_code(unsafe_code)
        print(f"安全限制结果:")
        print(json.dumps(unsafe_result, indent=2, ensure_ascii=False))
        
        print("\n✅ MCP 服务器功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    print("🚀 独立 MCP 服务器测试")
    
    success = await test_mcp_server_directly()
    
    if success:
        print("\n🎉 MCP 服务器功能正常！")
        print("\n📋 接下来可以:")
        print("1. 运行完整的集成测试: python examples/python_executor/test_python_executor.py")
        print("2. 启动 MCP 服务器: python -m verl.tools.mcp_services.python_executor_server")
    else:
        print("\n❌ MCP 服务器测试失败，请检查依赖和代码")


if __name__ == "__main__":
    asyncio.run(main())