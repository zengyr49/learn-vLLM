package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"
)

// 1. 定义大模型返回的数据结构
type ChatCompletionChunk struct {
	Usage *Usage `json:"usage,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// 2. 预编译正则表达式，用于提取 data: 后面的 JSON
var sseRegex = regexp.MustCompile(`^data:\s*(.*)`)

func main() {
	wrapper.SetCtx(
		"ai-billing-plugin",
		wrapper.ParseConfig(parseConfig),
		wrapper.ProcessResponseHeaders(onResponseHeaders),
		wrapper.ProcessResponseBody(onResponseBody),
	)
}

// 插件配置（暂留空，可扩展用于接收外部参数）
type PluginConfig struct{}

func parseConfig(json gjson.Result, config *PluginConfig) error {
	return nil
}

func onResponseHeaders(ctx wrapper.HttpContext, config PluginConfig) types.Action {
	return types.ActionContinue
}

// 3. 核心拦截与处理逻辑
func onResponseBody(ctx wrapper.HttpContext, config PluginConfig, body []byte) types.Action {
	// 按行切分数据流
	lines := bytes.Split(body, []byte("\n"))

	for _, line := range lines {
		lineStr := strings.TrimSpace(string(line))

		// 正则提取 JSON 部分
		matches := sseRegex.FindStringSubmatch(lineStr)
		if len(matches) > 1 {
			jsonStr := matches[1]

			// 忽略流结束标志
			if jsonStr == "[DONE]" {
				break
			}

			// 解析 JSON
			var chunk ChatCompletionChunk
			err := json.Unmarshal([]byte(jsonStr), &chunk)
			if err != nil {
				continue // 容错：跳过解析失败的数据块
			}

			// 提取计费信息（防空指针）
			if chunk.Usage != nil {
				promptTokens := chunk.Usage.PromptTokens
				completionTokens := chunk.Usage.CompletionTokens

				// 获取租户标识
				tenantId, _ := proxywasm.GetHttpRequestHeader("X-Tenant-Id")
				if tenantId == "" {
					tenantId = "default-tenant"
				}

				// 发起异步上报
				reportBillingAsync(tenantId, promptTokens, completionTokens)
			}
		}
	}

	// 必须返回 Continue，确保大模型的流式数据能顺利发给客户端
	return types.ActionContinue
}

// 4. 异步发送计费流水
func reportBillingAsync(tenantId string, prompt int, completion int) {
	payload := fmt.Sprintf(`{"tenant_id": "%s", "prompt_tokens": %d, "completion_tokens": %d}`,
		tenantId, prompt, completion)

	_, err := proxywasm.DispatchHttpCall(
		"billing-service-cluster", // 必须在 Envoy/Higress 中预先定义好的 Cluster 名称
		[][2]string{
			{":method", "POST"},
			{":path", "/api/v1/record"},
			{":authority", "billing.internal"},
			{"content-type", "application/json"},
		},
		[]byte(payload),
		nil,
		5000,
		func(numHeaders, bodySize, numTrailers int) {
			log.Infof("租户 %s 计费上报成功", tenantId)
		},
	)

	if err != nil {
		log.Errorf("异步上报计费请求失败: %v", err)
	}
}
