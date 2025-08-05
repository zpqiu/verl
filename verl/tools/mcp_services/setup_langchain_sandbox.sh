#!/bin/bash
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

# LangChain Sandbox MCP Server 安装脚本

set -e  # 遇到错误立即退出

echo "🚀 开始安装 LangChain Sandbox MCP Server 依赖..."

# 检查 Python 版本
echo "📋 检查 Python 版本..."
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python 版本: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ 错误: 需要 Python 3.10 或更高版本"
    exit 1
fi

echo "✅ Python 版本检查通过"

# 检查并安装 Deno
echo "📋 检查 Deno..."
if command -v deno &> /dev/null; then
    deno_version=$(deno --version | head -n1)
    echo "✅ Deno 已安装: $deno_version"
else
    echo "📦 安装 Deno..."
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Linux 或 macOS
        curl -fsSL https://deno.land/install.sh | sh
        echo "请将 Deno 添加到 PATH: export PATH=\"\$HOME/.deno/bin:\$PATH\""
        export PATH="$HOME/.deno/bin:$PATH"
    else
        echo "❌ 请手动安装 Deno: https://docs.deno.com/runtime/getting_started/installation/"
        exit 1
    fi
fi

# 安装 Python 依赖
echo "📦 安装 Python 依赖..."

# 安装 langchain-sandbox
echo "安装 langchain-sandbox..."
pip install langchain-sandbox

# 安装其他可能需要的依赖
echo "安装其他依赖..."
pip install fastmcp asyncio

echo "✅ Python 依赖安装完成"

# 验证安装
echo "🧪 验证安装..."
python3 -c "
try:
    from langchain_sandbox import PyodideSandbox
    print('✅ langchain-sandbox 导入成功')
except ImportError as e:
    print(f'❌ langchain-sandbox 导入失败: {e}')
    exit(1)

try:
    from fastmcp import FastMCP
    print('✅ fastmcp 导入成功')
except ImportError as e:
    print(f'❌ fastmcp 导入失败: {e}')
    exit(1)
"

echo "🎉 安装完成！"
echo ""
echo "📝 使用说明:"
echo "1. 启动服务器:"
echo "   python verl/tools/mcp_services/langchain_sandbox_server.py"
echo ""
echo "2. 运行测试:"
echo "   python verl/tools/mcp_services/test_langchain_sandbox.py"
echo ""
echo "3. 查看文档:"
echo "   cat verl/tools/mcp_services/README_langchain_sandbox.md"
echo ""
echo "🔧 如果遇到问题:"
echo "- 确保 Deno 在 PATH 中: export PATH=\"\$HOME/.deno/bin:\$PATH\""
echo "- 首次运行可能需要几秒钟下载 Pyodide"
echo "- 查看日志获取详细错误信息"