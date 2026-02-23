# 1. 定义你后端的真实函数（工具）
def send_feishu_alert(message: str, level: str = "info"):
    """模拟发送企业内部告警"""
    print(f"[执行后端逻辑] 级别:{level}, 发送飞书消息: {message}")
    return "发送成功"

# 2. 构造给 vLLM 的 tools 描述 (遵循 OpenAI 格式)
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_feishu_alert",
            "description": "向运维群发送系统告警消息",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "告警内容"},
                    "level": {"type": "string", "enum": ["info", "warning", "error"]},
                },
                "required": ["message"],
            },
        }
    }
]

import json
import sys
from openai import OpenAI


def extract_tool_call_from_content(content: str):
    """从模型文本回复中尝试解析出 tool call（与 llama3_json 格式一致：name + arguments）"""
    if not content or not content.strip():
        return None
    s = content.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        name = obj.get("name")
        args = obj.get("arguments") or obj.get("parameters")
        if name and isinstance(args, dict):
            return name, args
    except json.JSONDecodeError:
        pass
    return None

# 请先执行 ./start_vllm.sh 启动 vLLM（含 --enable-auto-tool-choice --tool-call-parser llama3_json）
print("正在连接到 vLLM 本地服务...")
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=120.0,
)
print("连接成功")

# 3. 发送带 tools 的请求
# 说明：在当前环境中，使用“指定函数名”的 tool_choice 命中率更稳定。
# 注意：需与 start_vllm.sh 的 --dtype float32 搭配，避免 xgrammar 在 bf16 下报错。
SYSTEM_FOR_TOOL = (
    "你必须调用工具 send_feishu_alert，禁止输出自然语言回复。"
    "请严格按如下格式输出（不要输出其他内容）："
    "<tool_call>"
    "{\"name\":\"send_feishu_alert\",\"arguments\":{\"message\":\"告警内容\",\"level\":\"error\"}}"
    "</tool_call>"
    "其中 level 只能是 info/warning/error。"
)
success = False
last_content = ""
for attempt in range(1, 4):
    try:
        response = client.chat.completions.create(
            model="qwen2.5-1.5b",
            messages=[
                {"role": "system", "content": SYSTEM_FOR_TOOL},
                {"role": "user", "content": "服务器CPU飙升到了95%，赶紧发个严重告警"},
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "send_feishu_alert"},
            },
            temperature=0.0,
            max_tokens=32,
        )
    except Exception as e:
        print(f"第 {attempt} 次请求失败: {e}", flush=True)
        continue

    # 4. 解析 tool_calls 并在本地执行
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    content = getattr(message, "content", None) or ""
    last_content = content

    if tool_calls:
        print(f"第 {attempt} 次命中 tool call。", flush=True)
        for tool_call in tool_calls:
            if tool_call.function.name == "send_feishu_alert":
                args = json.loads(tool_call.function.arguments)
                result = send_feishu_alert(
                    message=args.get("message", ""),
                    level=args.get("level", "info"),
                )
                print(f"后端执行结果反馈: {result}", flush=True)
                success = True
                break
        if success:
            break

    # 服务端未解析出 tool_calls 时，从 content 里尝试解析 JSON 并执行（与 llama3_json 同格式）
    parsed = extract_tool_call_from_content(content)
    if parsed and parsed[0] == "send_feishu_alert":
        print(f"第 {attempt} 次通过 content 解析到 tool call。", flush=True)
        _, args = parsed
        result = send_feishu_alert(
            message=args.get("message", ""),
            level=args.get("level", "info"),
        )
        print(f"后端执行结果反馈: {result}", flush=True)
        success = True
        break

    print(f"第 {attempt} 次未命中 tool call，继续重试。", flush=True)

if not success:
    print("连续重试后仍未调用工具，模型最后一次回复:", last_content or "<空>", flush=True)
