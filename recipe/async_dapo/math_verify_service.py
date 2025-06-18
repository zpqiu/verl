"""
Skywork Math Scoring Service (Fixed Version)

基于 skywork2.py 中的 compute_score 函数的 FastAPI 服务，修复了信号处理问题。
"""

import asyncio
import logging
import random
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import datasets
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    logging.warning("math-verify 库未安装，请运行: pip install math-verify")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储数据集
PROMPT_TO_ANSWER: Dict[str, str] = {}

def load_dataset():
    """加载数据集并构建prompt到answer的映射"""
    global PROMPT_TO_ANSWER
    try:
        logger.info("正在加载 Skywork-OR1-RL-Data-Math 数据集...")
        ds = datasets.load_dataset('pe-nlp/Skywork-OR1-RL-Data-Math', split='math')
        
        prompt2answer = {}
        for example in ds:
            prompt = example['prompt'][0]['content']
            ground_truth = example['ground_truth']
            prompt2answer[prompt] = ground_truth
        
        PROMPT_TO_ANSWER = prompt2answer
        logger.info(f"成功加载 {len(PROMPT_TO_ANSWER)} 个问答对")
        
        # 打印前5个QA对进行验证
        for i, (prompt, answer) in enumerate(list(PROMPT_TO_ANSWER.items())[:5]):
            logger.info(f"样例 {i}: {prompt[:50]}... -> {answer}")
            
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        PROMPT_TO_ANSWER = {}

# 创建 FastAPI 应用
app = FastAPI(
    title="Skywork Math Scoring Service (Fixed)",
    description="基于 Skywork2 的数学答案评分服务，修复了信号处理问题",
    version="1.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 应用启动时加载数据集
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化操作"""
    load_dataset()

# Pydantic 模型定义
class ScoreRequest(BaseModel):
    """评分请求模型"""
    data_source: Optional[str] = Field(None, description="数据源标识")
    solution_str: str = Field(..., description="模型生成的解答字符串")
    ground_truth: Optional[Union[str, List[str]]] = Field(None, description="标准答案（字符串或字符串列表），如果提供prompt则可为空")
    prompt: Optional[str] = Field(None, description="问题prompt，如果提供则会自动查找ground_truth")
    extra_info: Optional[Dict[str, Any]] = Field(None, description="额外信息")
    enable_debug: Optional[bool] = Field(False, description="是否启用调试日志")

class BatchScoreRequest(BaseModel):
    """批量评分请求模型"""
    items: List[ScoreRequest] = Field(..., description="评分项目列表")
    concurrent_limit: Optional[int] = Field(10, description="并发限制")

class ScoreResponse(BaseModel):
    """评分响应模型"""
    score: float = Field(..., description="评分结果 (1.0=正确, -1.0=错误)")
    acc: float = Field(..., description="准确率 (1.0=正确, 0.0=错误)")
    pred: str = Field(..., description="提取的预测答案")
    format_correct: bool = Field(..., description="格式是否正确")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    error: Optional[str] = Field(None, description="错误信息")

class BatchScoreResponse(BaseModel):
    """批量评分响应模型"""
    results: List[ScoreResponse] = Field(..., description="评分结果列表")
    total_count: int = Field(..., description="总数量")
    correct_count: int = Field(..., description="正确数量")
    accuracy: float = Field(..., description="总体准确率")
    total_processing_time: float = Field(..., description="总处理时间（秒）")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="时间戳")
    math_verify_available: bool = Field(..., description="math-verify 库是否可用")
    dataset_loaded: bool = Field(..., description="数据集是否已加载")
    dataset_size: int = Field(..., description="数据集大小")
    version: str = Field(..., description="服务版本")

# 核心函数（从 skywork2.py 移植）
def is_format_correct(completion: str) -> bool:
    """检查完成文本的格式是否正确"""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    if not re.match(pattern, completion, re.DOTALL):
        return False
    
    # 检查所有标签是否只出现一次
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    
    # 检查 <think>...</think> 是否为空
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, completion, re.DOTALL)
    if think_match and think_match.group(1).strip() == "":
        return False
    
    return True

def extract_answer_part(response: str) -> str:
    """从响应中提取答案部分"""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def safe_parse(text: str) -> Optional[Any]:
    """安全的解析函数，避免信号问题"""
    try:
        # 尝试使用默认的 parse 函数
        return parse(text)
    except ValueError as e:
        if "signal only works in main thread" in str(e):
            logger.warning(f"Signal error in parse, trying alternative approach for: {text}")
            # 如果是信号错误，返回 None，让调用者处理
            return None
        else:
            logger.warning(f"Parse failed for '{text}': {e}")
            return None
    except Exception as e:
        logger.warning(f"Parse failed for '{text}': {e}")
        return None

def safe_verify(pred_parsed: Any, gold_parsed: Any) -> bool:
    """安全的验证函数，避免信号问题"""
    try:
        return verify(gold_parsed, pred_parsed)
    except ValueError as e:
        if "signal only works in main thread" in str(e):
            logger.warning("Signal error in verify, skipping verification")
            return False
        else:
            logger.warning(f"Verify failed: {e}")
            return False
    except Exception as e:
        logger.warning(f"Verify failed: {e}")
        return False

def compute_score(data_source: Optional[str], solution_str: str, 
                 ground_truth: Union[str, List[str]], extra_info: Optional[Dict],
                 enable_debug: bool = False) -> Dict[str, Any]:
    """计算评分（从 skywork2.py 移植，修复信号问题）"""
    should_log = enable_debug or random.randint(0, 512) == 1
    result = None
    
    if should_log:
        logger.info("\n=== Skywork Scoring Debug Log ===")
        logger.info(f"Response: {solution_str}")
        logger.info(f"Ground Truth: {ground_truth}")

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    if not is_format_correct(solution_str.strip()):
        result = {
            "score": -1.0,
            "acc": 0.0,
            "pred": "[INVALID FORMAT]",
        }
        if should_log:
            logger.info("Invalid format")
            logger.info(f"Final Result: {result}")
            logger.info("================================\n")
        return result

    # 提取 <answer>...</answer>
    final_answer = extract_answer_part(solution_str)

    if final_answer == "":
        result = {
            "score": -1.0,
            "acc": 0.0,
            "pred": "[EMPTY ANSWER]",
        }
        if should_log:
            logger.info("Empty answer")
            logger.info(f"Final Result: {result}")
            logger.info("================================\n")
        return result

    # 尝试解析答案
    try:
        math_verify_parsed = safe_parse(final_answer)
        if math_verify_parsed is None:
            raise Exception("Parse returned None")
    except Exception:
        result = {
            "score": -1.0,
            "acc": 0.0,
            "pred": final_answer,  # 至少返回原始答案
        }
        if should_log:
            logger.info("Parsing failed")
            logger.info(f"Final Result: {result}")
            logger.info("================================\n")
        return result
    
    # 检查解析结果
    if not isinstance(math_verify_parsed, (list, tuple)) or len(math_verify_parsed) < 2:
        result = {
            "score": -1.0,
            "acc": 0.0,
            "pred": final_answer,  # 至少返回原始答案
        }
        if should_log:
            logger.info("Invalid parse result length")
            logger.info(f"Final Result: {result}")
            logger.info("================================\n")
        return result
    
    # 首先进行快速字符串匹配
    pred_answer = str(math_verify_parsed[1])
    if pred_answer in ground_truth:
        result = {
            "score": 1.0,
            "acc": 1.0,
            "pred": pred_answer,
        }
        if should_log:
            logger.info("Exact string match found")
            logger.info(f"Final Result: {result}")
            logger.info("================================\n")
        return result
    
    # 回退到语义验证
    for gt in ground_truth:
        try:
            if len(pred_answer) > 10 and len(pred_answer) > len(gt) * 5:
                logger.warning(f"Skip verification for {pred_answer}, gt: {gt}")
                continue

            # 解析标准答案
            gold_parsed = safe_parse(f"\\boxed{{{gt}}}")
            if gold_parsed is None:
                if should_log:
                    logger.info(f"Failed to parse ground truth: {gt}")
                continue

            if safe_verify(math_verify_parsed, gold_parsed):
                result = {
                    "score": 1.0,
                    "acc": 1.0,
                    "pred": pred_answer,
                }
                if should_log:
                    logger.info("Semantic verification succeeded")
                    logger.info(f"Final Result: {result}")
                    logger.info("================================\n")
                return result
        except Exception as e:
            if should_log:
                logger.info(f"Verification failed for ground truth: {gt}, error: {e}")
            continue
    
    # 经过上述匹配后很不可能是正确的
    result = {
        "score": -1.0,
        "acc": 0.0,
        "pred": pred_answer,
    }
    if should_log:
        logger.info("No matches found")
        logger.info(f"Final Result: {result}")
        logger.info("================================\n")
    return result

async def score_single_answer(request: ScoreRequest) -> ScoreResponse:
    """评分单个答案"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not MATH_VERIFY_AVAILABLE:
            raise HTTPException(status_code=500, detail="math-verify 库未安装")
        
        # 确定ground_truth
        ground_truth = request.ground_truth
        if ground_truth is None and request.prompt is not None:
            # 从数据集中查找ground_truth
            ground_truth = PROMPT_TO_ANSWER.get(request.prompt)
            if ground_truth is None:
                return ScoreResponse(
                    score=-1.0,
                    acc=0.0,
                    pred="[PROMPT NOT FOUND]",
                    format_correct=False,
                    error="未找到对应的ground_truth",
                    processing_time=asyncio.get_event_loop().time() - start_time
                )
        elif ground_truth is None:
            return ScoreResponse(
                score=-1.0,
                acc=0.0,
                pred="[NO GROUND TRUTH]",
                format_correct=False,
                error="必须提供ground_truth或prompt",
                processing_time=asyncio.get_event_loop().time() - start_time
            )
        
        # 检查格式
        format_correct = is_format_correct(request.solution_str)
        
        # 直接在主线程中运行评分，避免信号问题
        score_result = compute_score(
            request.data_source,
            request.solution_str,
            ground_truth,
            request.extra_info,
            request.enable_debug
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ScoreResponse(
            score=score_result["score"],
            acc=score_result["acc"],
            pred=score_result["pred"],
            format_correct=format_correct,
            processing_time=processing_time
        )
    
    except Exception as e:
        processing_time = asyncio.get_event_loop().time() - start_time
        error_msg = f"评分失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return ScoreResponse(
            score=-1.0,
            acc=0.0,
            pred="",
            format_correct=False,
            error=error_msg,
            processing_time=processing_time
        )



# API 端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        math_verify_available=MATH_VERIFY_AVAILABLE,
        dataset_loaded=bool(PROMPT_TO_ANSWER),
        dataset_size=len(PROMPT_TO_ANSWER),
        version="1.0.1"
    )

@app.post("/score", response_model=ScoreResponse)
async def score_answer(request: ScoreRequest):
    """
    评分单个数学答案
    
    支持 Skywork2 格式的答案评分：
    - 格式验证（<think>...</think><answer>...</answer>）
    - 答案提取
    - 语义验证
    """
    return await score_single_answer(request)

@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_answers_batch(request: BatchScoreRequest):
    """
    批量评分数学答案
    
    支持并发处理多个答案评分请求
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # 限制并发数量
        semaphore = asyncio.Semaphore(request.concurrent_limit)
        
        async def score_with_semaphore(score_request: ScoreRequest):
            async with semaphore:
                return await score_single_answer(score_request)
        
        # 并发执行评分
        tasks = [score_with_semaphore(item) for item in request.items]
        results = await asyncio.gather(*tasks)
        
        # 计算统计信息
        total_count = len(results)
        correct_count = sum(1 for result in results if result.acc > 0)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        total_processing_time = asyncio.get_event_loop().time() - start_time
        
        return BatchScoreResponse(
            results=results,
            total_count=total_count,
            correct_count=correct_count,
            accuracy=accuracy,
            total_processing_time=total_processing_time
        )
    
    except Exception as e:
        error_msg = f"批量评分失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)



@app.get("/")
async def root():
    """根端点，返回服务信息"""
    return {
        "message": "Skywork Math Scoring Service (Fixed)",
        "description": "基于 Skywork2 的数学答案评分服务，修复了信号处理问题",
        "version": "1.0.1",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "math_verify_service:app",
        host="0.0.0.0",
        port=8000,
        # reload=False,
        log_level="info"
    ) 