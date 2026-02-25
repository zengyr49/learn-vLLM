# Higress WASM 插件构建（需 Go 1.24+，原生 wasip1/wasm 支持）
.PHONY: build build-wasm clean

build:
	go build -o /dev/null ./

# 产出 main.wasm，供 Higress WasmPlugin 通过 url: oci://... 或本地挂载使用
build-wasm:
	GOOS=wasip1 GOARCH=wasm go build -buildmode=c-shared -o main.wasm ./

clean:
	rm -f main.wasm main.h
