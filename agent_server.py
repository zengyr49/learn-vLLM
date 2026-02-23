import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

app = FastAPI(title="AI Article Summarizer Agent")


def extract_json_from_text(text: str) -> str:
    """从模型输出中提取 JSON 字符串（可能被 markdown 或前后文字包裹）。"""
    if not text or not text.strip():
        raise ValueError("模型返回内容为空")
    s = text.strip()
    # 去掉 ```json ... ``` 或 ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    # 取第一个 { 到最后一个 } 之间的内容
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    return s

# ==========================================
# 1. 定义数据结构 (强类型约束，对接上下游)
# ==========================================
class ArticleRequest(BaseModel):
    content: str = Field(..., description="需要分析的文章原文或长文本")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="文章的200字核心摘要")
    tags: list[str] = Field(..., description="提取的3-5个技术标签")
    sentiment: str = Field(..., description="文章情感倾向：正面/中立/负面")

# ==========================================
# 2. 封装 LLM 异步调用客户端 (对接 vLLM)
# ==========================================
class VLLMClient:
    def __init__(self):
        # 指向你本地的 vLLM 服务
        self.base_url = "http://localhost:8000/v1"
        self.model_name = "qwen2.5-1.5b" # 替换为你实际运行的模型名称

    async def generate_json_summary(self, text: str) -> dict:
        """调用大模型，并强制要求返回结构化 JSON"""
        
        # 提示词工程 (Prompt Engineering) 的核心：设定清晰的边界和输出格式
        system_prompt = """
        你是一个资深的技术编辑 Agent。你的任务是分析用户输入的文章。
        请严格以 JSON 格式输出，包含以下三个字段：
        1. "summary": 文章的简短摘要。
        2. "tags": 3到5个关键技术标签（数组）。
        3. "sentiment": 情感倾向（"正面", "中立", 或 "负面"）。
        不要输出任何额外的解释文本，只输出合法的 JSON 字符串。
        """

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析以下文章：\n{text}"}
            ],
            "temperature": 0.1,  # 降低温度，让模型的输出更稳定、更确定
            # 不使用 response_format.json_object：在 CPU/V1 下会触发结构化输出后端导致 vLLM 崩溃，
            # 仅靠 prompt 约束输出 JSON，由下方 JSONDecodeError 做容错即可。
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                result_data = response.json()
                content = result_data.get("choices", [{}])[0].get("message", {}).get("content") or ""

                # 从可能被 markdown/前后文字包裹的文本中提取并解析 JSON
                json_str = extract_json_from_text(content)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败，原始内容片段: {repr(content[:500]) if content else 'None'}")
                raise e
            except Exception as e:
                print(f"大模型调用失败: {str(e)}")
                raise

llm_client = VLLMClient()

# ==========================================
# 3. 暴露 FastAPI 路由 (对接前端/内部其他微服务)
# ==========================================
@app.post("/api/v1/analyze-article", response_model=SummaryResponse)
async def analyze_article(req: ArticleRequest):
    try:
        # 1. 业务逻辑校验（例如判断文本长度）
        if len(req.content) < 50:
            raise HTTPException(status_code=400, detail="文章内容太短，无法分析。")
            
        # 2. 异步调用大模型进行处理
        raw_result = await llm_client.generate_json_summary(req.content)
        
        # 3. 结构化返回（如果大模型没按规矩返回，Pydantic 会在这里报错拦截）
        return SummaryResponse(**raw_result)
        
    except json.JSONDecodeError:
        # 容灾处理：如果大模型“幻觉”发作，没有返回合法 JSON 的处理机制
        raise HTTPException(status_code=500, detail="AI 模型返回格式错误，未能解析为 JSON。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)