# test_client.py
from openai import OpenAI

# 将 base_url 指向你刚刚启动的 vLLM 本地服务
print("正在连接到 vLLM 本地服务...")
client = OpenAI(
    api_key="EMPTY", # 本地 vLLM 默认不需要鉴权
    base_url="http://localhost:8000/v1",
)
print("连接成功")

print("正在发送请求...")
response = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[
        {"role": "system", "content": "你是一个资深的云原生架构师。"},
        {"role": "user", "content": "请用一句话解释什么是高可用架构？"}
    ]
)
print("请求完成")
print("响应内容:")
print(response.choices[0].message.content)
print("请求完成")




