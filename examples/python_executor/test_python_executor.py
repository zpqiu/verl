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
Python 代码执行工具的测试脚本

这个脚本演示了如何使用 Python 代码执行工具，
包括基础功能测试和集成测试。
"""

import asyncio
import json
import logging
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from verl.tools.utils.tool_registry import initialize_tools_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_python_execution():
    """测试基础的 Python 代码执行功能"""
    print("🔍 开始测试基础 Python 代码执行功能...")
    
    # 初始化工具
    config_path = "examples/python_executor/config/python_executor_tool_config.yaml"
    tools = initialize_tools_from_config(config_path)
    
    if not tools:
        print("❌ 工具初始化失败")
        return False
    
    print(f"✅ 成功初始化 {len(tools)} 个工具")
    
    # 获取 Python 执行工具
    executor_tool = None
    for tool in tools:
        if "execute_python_code" in tool.name:
            executor_tool = tool
            break
    
    if not executor_tool:
        print("❌ 未找到 Python 代码执行工具")
        return False
    
    print(f"✅ 找到工具: {executor_tool.name}")
    
    # 创建工具实例
    instance_id = await executor_tool.create()
    print(f"✅ 创建工具实例: {instance_id}")
    
    # 测试用例
    test_cases = [
        {
            "name": "简单计算",
            "code": '''
result = 2 + 3
print(f"2 + 3 = {result}")
            '''.strip()
        },
        {
            "name": "数学库使用",
            "code": '''
import math
print(f"π = {math.pi:.4f}")
print(f"sin(π/2) = {math.sin(math.pi/2)}")
print(f"sqrt(16) = {math.sqrt(16)}")
            '''.strip()
        },
        {
            "name": "循环和列表",
            "code": '''
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(f"原数组: {numbers}")
print(f"平方数组: {squares}")
print(f"总和: {sum(squares)}")
            '''.strip()
        },
        {
            "name": "错误处理测试",
            "code": '''
# 这会产生一个错误
result = 1 / 0
            '''.strip()
        },
        {
            "name": "安全限制测试",
            "code": '''
# 这应该被拒绝
import os
os.system("ls")
            '''.strip()
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        print(f"代码:\n{test_case['code']}")
        
        try:
            # 执行代码
            result, reward, metrics = await executor_tool.execute(
                instance_id=instance_id,
                parameters={"code": test_case['code']}
            )
            
            print(f"执行结果:\n{result}")
            print(f"奖励分数: {reward}")
            print(f"指标: {json.dumps(metrics, indent=2)}")
            
            # 检查是否成功（对于安全限制测试，失败是预期的）
            if test_case['name'] == "安全限制测试":
                if "禁止使用" in result:
                    print("✅ 安全限制正常工作")
                    success_count += 1
                else:
                    print("❌ 安全限制未生效")
            elif test_case['name'] == "错误处理测试":
                if "除零" in result or "division by zero" in result:
                    print("✅ 错误处理正常")
                    success_count += 1
                else:
                    print("❌ 错误处理异常")
            else:
                if "✅" in result:
                    print("✅ 执行成功")
                    success_count += 1
                else:
                    print("❌ 执行失败")
                    
        except Exception as e:
            print(f"❌ 测试出错: {e}")
    
    # 清理
    await executor_tool.release(instance_id)
    print(f"\n📊 测试完成: {success_count}/{len(test_cases)} 个测试用例通过")
    
    return success_count == len(test_cases)


async def test_available_modules():
    """测试获取可用模块功能"""
    print("\n🔍 测试获取可用模块功能...")
    
    config_path = "examples/python_executor/config/python_executor_tool_config.yaml"
    tools = initialize_tools_from_config(config_path)
    
    # 获取模块查询工具
    modules_tool = None
    for tool in tools:
        if "get_available_modules" in tool.name:
            modules_tool = tool
            break
    
    if not modules_tool:
        print("❌ 未找到模块查询工具")
        return False
    
    print(f"✅ 找到工具: {modules_tool.name}")
    
    # 创建实例并执行
    instance_id = await modules_tool.create()
    
    try:
        result, reward, metrics = await modules_tool.execute(
            instance_id=instance_id,
            parameters={}
        )
        
        print("可用模块信息:")
        print(result)
        
        await modules_tool.release(instance_id)
        return True
        
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return False


def create_training_config_example():
    """创建训练配置示例"""
    print("\n📝 创建训练配置示例...")
    
    training_config = {
        "rollout": {
            "tools_config_file": "examples/python_executor/config/python_executor_tool_config.yaml",
            "other_config": "..."
        },
        "model": {
            "model_name": "your_model",
            "other_config": "..."
        }
    }
    
    with open("examples/python_executor/training_config_example.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ 训练配置示例已保存到 examples/python_executor/training_config_example.yaml")


async def main():
    """主测试函数"""
    print("🚀 开始 Python 代码执行工具集成测试")
    
    try:
        # 测试基础功能
        basic_test_passed = await test_basic_python_execution()
        
        # 测试模块查询
        modules_test_passed = await test_available_modules()
        
        # 创建配置示例
        create_training_config_example()
        
        # 总结
        print(f"\n📊 测试总结:")
        print(f"  基础功能测试: {'✅ 通过' if basic_test_passed else '❌ 失败'}")
        print(f"  模块查询测试: {'✅ 通过' if modules_test_passed else '❌ 失败'}")
        
        if basic_test_passed and modules_test_passed:
            print("\n🎉 所有测试通过！Python 代码执行工具已准备就绪。")
            print("\n📋 下一步操作:")
            print("1. 将工具配置添加到您的训练配置文件中")
            print("2. 在训练脚本中指定 tools_config_file 路径")
            print("3. 模型就可以使用 Python 代码执行功能了")
        else:
            print("\n❌ 部分测试失败，请检查配置和依赖")
            
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置环境变量
    os.environ["VERL_LOGGING_LEVEL"] = "INFO"
    
    # 运行测试
    asyncio.run(main())