# Python 代码执行工具

这是一个基于 MCP (Model Context Protocol) 的 Python 代码执行工具，允许在 verl 训练流程中安全地执行 Python 代码。

## 功能特性

- **安全执行环境**: 限制危险操作，如文件系统访问、子进程执行等
- **输出捕获**: 捕获标准输出和错误输出
- **超时控制**: 防止代码无限循环
- **常用库支持**: 支持 math, json, datetime, random, numpy, pandas 等常用库

## 安装依赖

首先安装必要的依赖：

```bash
pip install fastmcp
```

如果需要使用数值计算功能，可选安装：

```bash
pip install numpy pandas
```

## 使用方法

### 1. 启动 MCP 服务

服务会自动通过配置文件启动，也可以手动启动进行测试：

```bash
python -m verl.tools.mcp_services.python_executor_server --host 127.0.0.1 --port 8001
```

### 2. 配置文件说明

#### MCP 服务器配置 (`mcp_server.json`)

```json
{
    "mcpServers": {
        "Python Executor": {
            "command": "python", 
            "args": ["-m", "verl.tools.mcp_services.python_executor_server", "--host", "127.0.0.1", "--port", "8001"],
            "env": {
                "PYTHONPATH": "."
            }
        }
    }
}
```

#### 工具配置 (`python_executor_tool_config.yaml`)

```yaml
tools:
  - class_name: verl.tools.python_executor_tool.PythonExecutorTool
    config:
      rate_limit: 60  # 每分钟最多60次请求
      timeout: 60     # 60秒超时
      type: mcp
    mcp:
      mcp_servers_config_path: ./examples/python_executor/config/mcp_server.json
      tool_selected_list:
        - execute_python_code
        - get_available_modules
```

### 3. 在 verl 训练中使用

将工具配置添加到您的训练配置中：

```yaml
# 在您的训练配置文件中
rollout:
  tools_config_file: ./examples/python_executor/config/python_executor_tool_config.yaml
```

## 可用工具

### execute_python_code

执行 Python 代码并返回结果。

**参数:**
- `code` (str): 要执行的 Python 代码
- `timeout` (int, 可选): 执行超时时间（秒），默认30秒

**返回:**
- 执行结果，包含成功状态、输出、错误信息等

**示例:**
```python
# 基础计算
result = 2 + 3
print(f"2 + 3 = {result}")

# 使用数学库
import math
print(f"π = {math.pi}")
print(f"sin(π/2) = {math.sin(math.pi/2)}")

# 数据处理（如果安装了 numpy）
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr}")
print(f"平均值: {np.mean(arr)}")
```

### get_available_modules

获取可用模块和函数的列表。

**返回:**
- 可用模块及其功能的详细说明

## 安全限制

为了确保安全，以下操作被禁止：

- 文件系统操作 (`open`, `file`)
- 系统命令执行 (`os`, `subprocess`, `sys`)
- 动态导入 (`importlib`, `__import__`)
- 危险函数 (`exec`, `eval`)
- 用户输入 (`input`, `raw_input`)

## 集群环境部署

### ⚠️ 分布式训练注意事项

在分布式/集群环境中，多个 worker 进程同时启动可能导致端口冲突。请参考以下解决方案：

#### 推荐：STDIO 传输模式

修改 `mcp_server.json` 使用 STDIO 传输：

```json
{
    "mcpServers": {
        "Python Executor": {
            "command": "python", 
            "args": ["-m", "verl.tools.mcp_services.python_executor_server"],
            "env": {
                "PYTHONPATH": "."
            }
        }
    }
}
```

#### 备选：动态端口分配

如果需要网络访问，使用动态端口：

```json
{
    "mcpServers": {
        "Python Executor": {
            "command": "python", 
            "args": ["-m", "verl.tools.mcp_services.python_executor_server_with_port_finder", "--strategy", "pid"],
            "env": {
                "PYTHONPATH": "."
            }
        }
    }
}
```

### 集群部署测试

验证多进程启动：

```bash
python examples/python_executor/test_cluster_deployment.py
```

详细部署指南请参考：[CLUSTER_DEPLOYMENT.md](./CLUSTER_DEPLOYMENT.md)

## 故障排除

### 常见问题

1. **"Client failed to connect: Connection closed"**
   - ✅ 使用 STDIO 传输模式（推荐）
   - ✅ 检查 fastmcp 版本：`pip install --upgrade fastmcp`

2. **"Address already in use"**
   - ✅ 使用动态端口分配
   - ✅ 改用 STDIO 传输模式

3. **代码执行被拒绝**
   - 检查代码是否包含被禁止的函数或模块
   - 确保代码语法正确

4. **连接超时**
   - 检查网络连接
   - 增加超时时间设置

### 日志调试

设置环境变量启用详细日志：

```bash
export VERL_LOGGING_LEVEL=DEBUG
```

### 多进程环境验证

```bash
# 验证不会重复启动服务
python examples/python_executor/verify_no_duplicate_startup.py

# 测试集群部署
python examples/python_executor/test_cluster_deployment.py
```

## 扩展开发

如果需要添加新的功能或修改解析逻辑，可以：

1. 扩展 MCP 服务器 (`python_executor_server.py`) 添加新工具
2. 修改工具类 (`python_executor_tool.py`) 的 `_parse_tool_result` 方法自定义结果解析
3. 更新配置文件以包含新的工具