// Runs the language model off the main thread so the page stays responsive while
// weights download, shaders compile and tokens are generated.
import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

let generator = null;
const stopper = new InterruptableStoppingCriteria();

const post = (type, data = {}) => self.postMessage({ type, ...data });

async function load({ model, dtype }) {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
  if (!(await navigator.gpu.requestAdapter())) throw new Error("No WebGPU adapter was found.");

  await generator?.dispose?.();
  generator = null;

  generator = await pipeline("text-generation", model, {
    device: "webgpu",
    dtype,
    progress_callback: (p) => post("progress", { progress: p }),
  });

  // One short generation compiles the GPU pipelines, so the first real answer is quick.
  post("status", { text: "Compiling shaders…" });
  await generator([{ role: "user", content: "Hi" }], { max_new_tokens: 1 });
  post("ready");
}

async function generate({ messages, options }) {
  if (!generator) throw new Error("Load a model first.");
  stopper.reset();

  let tokens = 0;
  let firstTokenAt = 0;
  const start = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    // Keep special tokens so reasoning/answer channels (e.g. gpt-oss) can be told apart.
    skip_special_tokens: false,
    callback_function: (text) => post("token", { text }),
    token_callback_function: () => {
      tokens++;
      if (!firstTokenAt) firstTokenAt = performance.now();
      const secs = (performance.now() - firstTokenAt) / 1000;
      if (tokens > 1 && secs > 0) post("stats", { tokens, tps: (tokens - 1) / secs });
    },
  });

  await generator(messages, {
    max_new_tokens: options.maxNewTokens,
    do_sample: options.temperature > 0,
    ...(options.temperature > 0 ? { temperature: options.temperature, top_p: 0.95 } : {}),
    streamer,
    stopping_criteria: stopper,
  });

  post("done", {
    tokens,
    ttft: firstTokenAt ? firstTokenAt - start : 0,
    interrupted: stopper.interrupted,
  });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "generate") await generate(data);
    else if (data.type === "interrupt") stopper.interrupt();
  } catch (err) {
    post("error", { message: err?.message ?? String(err), during: data.type });
  }
});
