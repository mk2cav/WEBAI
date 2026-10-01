# WebGPU MoE demo

Two pages, both running on WebGPU in the browser:

- **`index.html` (kernel demo)** runs the single `com.microsoft.MoE` kernel and checks its output.
- **`chat.html` (chat)** loads a complete Mixture-of-Experts language model and lets you ask it
  questions.

## Kernel demo

A browser demo for the [`com.microsoft.MoE`](https://huggingface.co/kernels/webgpu-kernels/com.microsoft.MoE)
WebGPU kernel from the Hugging Face Hub. The page loads the kernel with
[`@huggingface/kernels`](https://www.npmjs.com/package/@huggingface/kernels), runs a random
Mixture-of-Experts layer on your GPU, and compares the output with a CPU reference.

## Run it

```sh
npm start          # or: python3 -m http.server 8080
```

Open <http://localhost:8080> in a browser that supports WebGPU, such as a recent Chrome, Edge or
Safari. Then click **Load kernel** and **Run & verify**. The page must be able to reach
`huggingface.co`, because the kernel is downloaded from the Hub at load time.

## What the page shows

- **Configuration**: tokens, hidden size, expert inner size, number of experts, top-k,
  activation (`relu`, `gelu`, `silu`, `identity`), optional gated FFN (`fc3`), biases,
  top-k renormalisation, and a seed so runs can be reproduced.
- **Routing**: a heatmap of each token's softmax probability per expert, with the top-k
  experts outlined, and a bar chart of how many tokens each expert receives.
- **Result**: output shape, GPU and CPU timings, maximum absolute and relative error, and
  sample values from the GPU and the CPU side by side.
- **Benchmark**: the mean time per call over 20 runs, including upload and readback.
- **Kernel manifest**: the inputs, outputs and attribute defaults the kernel declares.

## Notes

- The demo reads `build/webgpu/manifest.json` to map ONNX input names (`input`,
  `router_probs`, `fc1_experts_weights`, …) to the kernel's own tensor names, and to choose
  float32 or float16. If the manifest can't be read, it falls back to ONNX Runtime's names.
- ONNX Runtime has used two layouts for expert weights: `[E, out, in]` in current releases and
  `[E, in, out]` in older ones. The demo tries the current layout first, falls back to the
  older one if the kernel rejects it, and checks the output against the matching reference.
- `vendor/huggingface-kernels/` contains `@huggingface/kernels@0.0.1-preview.3`, copied
  unmodified from npm under the Apache-2.0 license, so the demo does not depend on a CDN.

## Chat page

`chat.html` uses [transformers.js](https://www.npmjs.com/package/@huggingface/transformers) v4
(loaded from jsDelivr) to download an ONNX model from the Hub and run it on WebGPU in a Web
Worker. Replies stream in token by token.

| Preset | Type | Download |
| --- | --- | --- |
| `onnx-community/LFM2-8B-A1B-ONNX` | MoE, 8.3B total / about 1.5B active | several GB |
| `onnx-community/gpt-oss-20b-ONNX` | MoE, 21B total / about 3.6B active, reasoning | about 12 GB |
| `onnx-community/Qwen3-0.6B-ONNX` | dense, small (quick check that the setup works) | about 0.5 GB |

You can also enter any other transformers.js text-generation model ID. The page also:

- has a precision selector (`q4f16`, `q4` or `fp16`), with a progress bar for each downloaded file;
- keeps multi-turn history, and has an optional system prompt and settings for max new tokens
  and temperature;
- has a **Stop** button that interrupts generation;
- shows reasoning (`<think>…</think>` or gpt-oss's analysis channel) in a collapsible "Reasoning"
  section above the answer, and keeps it out of the chat history;
- shows tokens per second and the time to the first token.

The browser caches the weights, so only the first load downloads them. The models are large:
start with Qwen3 0.6B to check that your browser and GPU work, then try a MoE model.

## Layout

```
index.html               kernel demo page
chat.html                chat page
src/main.js              kernel loading, input construction, UI
src/chat.js              chat UI
src/chat-worker.js       model loading and generation (Web Worker)
src/chat-format.js       splits reasoning from the answer
src/moe-reference.js     CPU reference implementation of com.microsoft.MoE
src/style.css            styles (light and dark)
test/                    node:test tests (npm test)
vendor/                  vendored @huggingface/kernels
```
