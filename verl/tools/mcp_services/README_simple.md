# 简单 Python 沙盒 MCP Server

这是一个极简版本的 Python 代码执行沙盒，只提供一个基本的代码执行工具。

## 特点

- 🔒 **安全隔离**: 使用 WebAssembly 沙盒技术
- 🚀 **简单易用**: 只有一个工具 `execute_python`
- ⚡ **轻量级**: 最小化的依赖和配置

## 安装

```bash
# 安装依赖
pip install langchain-sandbox

# 如果需要 Deno (首次运行时会自动下载)
# 访问: https://docs.deno.com/runtime/getting_started/installation/
```

## 使用

### 启动服务器

```bash
# STDIO 模式
python verl/tools/mcp_services/simple_sandbox_server.py

# HTTP 模式  
python verl/tools/mcp_services/simple_sandbox_server.py --port 8080
```

### 运行测试

```bash
python verl/tools/mcp_services/test_simple_sandbox.py
```

## 工具

只有一个工具：

### `execute_python(code: str)`

在安全沙盒中执行 Python 代码。

**参数:**
- `code`: 要执行的 Python 代码字符串

**返回:**
```json
{
    "success": true/false,
    "result": "执行结果",
    "stdout": "标准输出",
    "stderr": "错误输出"
}
```

## 示例

### 基本计算
```python
execute_python("print('Hello!'); 2 + 2")
# 返回: {"success": true, "result": 4, "stdout": "Hello!\n", "stderr": ""}
```

### 数学运算
```python
execute_python("import math; math.sqrt(16)")
# 返回: {"success": true, "result": 4.0, "stdout": "", "stderr": ""}
```

### 错误处理
```python
execute_python("1 / 0")
# 返回: {"success": false, "result": null, "stdout": "", "stderr": "...ZeroDivisionError..."}
```

## 限制

- 每次执行都是独立的（无状态）
- 只能使用内置库和部分常用包
- 无法持久化文件
- 无法访问网络（除非包需要）

## 与复杂版本对比

| 特性 | 简单版本 | 复杂版本 |
|------|----------|----------|
| 工具数量 | 1个 | 6个 |
| 会话管理 | 无 | 多会话 |
| 状态保持 | 无 | 支持 |
| 可视化 | 无 | 自动捕获 |
| 包管理 | 自动 | 手动指定 |
| 复杂度 | 低 | 高 |

这个简单版本适合只需要基本 Python 代码执行功能的场景。