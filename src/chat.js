import { parseOutput } from "./chat-format.js";

// Mixture-of-Experts models first, then a small dense model for quick tests on modest GPUs.
const PRESETS = [
  {
    id: "onnx-community/LFM2-8B-A1B-ONNX",
    label: "LFM2 8B-A1B (MoE, about 1.5B active)",
    dtype: "q4",
    note: "Liquid AI's on-device MoE: 8.3B parameters in total, about 1.5B active per token. Downloads several GB.",
  },
  {
    id: "onnx-community/gpt-oss-20b-ONNX",
    label: "gpt-oss 20B (MoE, about 3.6B active)",
    dtype: "q4f16",
    note: "OpenAI's open-weight MoE reasoning model. Downloads about 12 GB and needs a high-end GPU with plenty of memory.",
  },
  {
    id: "onnx-community/Qwen3-0.6B-ONNX",
    label: "Qwen3 0.6B (dense, small)",
    dtype: "q4f16",
    note: "Not a MoE model. Small and fast, so it's a good way to check that your browser and GPU work.",
  },
  { id: "custom", label: "Custom model ID…", dtype: "q4f16", note: "Any transformers.js text-generation model in ONNX format." },
];

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./chat-worker.js", import.meta.url), { type: "module" });

const state = {
  ready: false,
  busy: false,
  history: [], // { role, content } with final answers only
  current: null, // { raw, el } for the reply being streamed
  files: new Map(), // download progress per file
};

// ---------------------------------------------------------------- status

function setStatus(text, kind) {
  $("status").textContent = text;
  $("status").dataset.kind = kind;
}

function setBusy(busy) {
  state.busy = busy;
  $("send").hidden = busy;
  $("stop").hidden = !busy;
  $("prompt").disabled = !state.ready;
  $("send").disabled = !state.ready;
  for (const chip of document.querySelectorAll(".chip")) chip.disabled = !state.ready || busy;
}

// ---------------------------------------------------------------- model selection

function initPresets() {
  $("preset").innerHTML = PRESETS.map((p, i) => `<option value="${i}">${p.label}</option>`).join("");
  $("preset").addEventListener("change", applyPreset);
  applyPreset();
}

function applyPreset() {
  const p = PRESETS[$("preset").value];
  $("custom-wrap").hidden = p.id !== "custom";
  $("dtype").value = p.dtype;
  $("preset-note").textContent = p.note;
}

function selectedModel() {
  const p = PRESETS[$("preset").value];
  return p.id === "custom" ? $("custom").value.trim() : p.id;
}

function loadModel() {
  const model = selectedModel();
  if (!model) {
    setStatus("Enter a model ID first", "warn");
    return;
  }
  state.ready = false;
  state.files.clear();
  $("progress").innerHTML = "";
  $("load").disabled = true;
  setBusy(false);
  setStatus(`Downloading ${model}…`, "busy");
  worker.postMessage({ type: "load", model, dtype: $("dtype").value });
}

function renderProgress() {
  const fmt = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(0)} MB`);
  $("progress").innerHTML = [...state.files.entries()]
    .map(([file, f]) => `
      <div class="file-progress">
        <span class="file-name" title="${file}">${file}</span>
        <span class="bar-track"><span class="bar" style="width:${f.progress.toFixed(1)}%"></span></span>
        <span class="file-size">${f.total ? `${fmt(f.loaded)} / ${fmt(f.total)}` : `${f.progress.toFixed(0)}%`}</span>
      </div>`)
    .join("");
}

// ---------------------------------------------------------------- messages

function addMessage(role, text = "") {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg msg-${role}`;
  if (role === "assistant") {
    el.innerHTML = `
      <details class="reasoning" hidden><summary>Reasoning</summary><div class="reasoning-text"></div></details>
      <div class="msg-text"></div>`;
    el.querySelector(".msg-text").textContent = text;
  } else {
    el.textContent = text;
  }
  $("messages").append(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

function renderReply() {
  const { raw, el } = state.current;
  const { reasoning, answer, thinking } = parseOutput(raw);
  const box = el.querySelector(".reasoning");
  box.hidden = !reasoning;
  box.querySelector("summary").textContent = thinking ? "Thinking…" : "Reasoning";
  box.querySelector(".reasoning-text").textContent = reasoning;
  el.querySelector(".msg-text").textContent = answer;
  el.classList.toggle("pending", !answer);
  const list = $("messages");
  if (list.scrollHeight - list.scrollTop - list.clientHeight < 80) list.scrollTop = list.scrollHeight;
}

function send(text) {
  text = text.trim();
  if (!text || !state.ready || state.busy) return;
  $("prompt").value = "";
  $("suggestions").hidden = true;
  addMessage("user", text);
  state.history.push({ role: "user", content: text });
  state.current = { raw: "", el: addMessage("assistant") };
  state.current.el.classList.add("pending");

  const system = $("system").value.trim();
  const messages = system ? [{ role: "system", content: system }, ...state.history] : [...state.history];
  setBusy(true);
  setStatus("Generating…", "busy");
  $("stats").textContent = "";
  worker.postMessage({
    type: "generate",
    messages,
    options: {
      maxNewTokens: Number($("max-tokens").value) || 1024,
      temperature: Math.max(0, Number($("temperature").value) || 0),
    },
  });
}

// ---------------------------------------------------------------- worker events

worker.addEventListener("message", ({ data }) => {
  switch (data.type) {
    case "progress": {
      const p = data.progress;
      if (p.status === "progress" && p.file) {
        state.files.set(p.file, { progress: p.progress ?? 0, loaded: p.loaded ?? 0, total: p.total ?? 0 });
        renderProgress();
      } else if (p.status === "done" && p.file && state.files.has(p.file)) {
        state.files.get(p.file).progress = 100;
        renderProgress();
      }
      break;
    }
    case "status":
      setStatus(data.text, "busy");
      break;
    case "ready":
      state.ready = true;
      $("load").disabled = false;
      $("load").textContent = "Load a different model";
      setStatus(`Ready: ${selectedModel()}`, "ok");
      setBusy(false);
      $("prompt").focus();
      break;
    case "token":
      if (!state.current) break;
      state.current.raw += data.text;
      renderReply();
      break;
    case "stats":
      $("stats").textContent = `${data.tokens} tokens · ${data.tps.toFixed(1)} tokens/s`;
      break;
    case "done": {
      const { answer } = parseOutput(state.current.raw);
      state.current.el.classList.remove("pending");
      if (!answer) state.current.el.querySelector(".msg-text").textContent = data.interrupted ? "(stopped)" : "(no answer)";
      state.history.push({ role: "assistant", content: answer });
      state.current = null;
      const ttft = data.ttft ? ` · first token after ${(data.ttft / 1000).toFixed(1)} s` : "";
      $("stats").textContent += ttft;
      setStatus(data.interrupted ? "Stopped" : `Ready: ${selectedModel()}`, "ok");
      setBusy(false);
      $("prompt").focus();
      break;
    }
    case "error":
      if (data.during === "load") {
        $("load").disabled = false;
        setStatus("Could not load the model", "error");
      } else {
        setStatus("Generation failed", "error");
        if (state.current) {
          state.current.el.classList.remove("pending");
          state.current.el.querySelector(".msg-text").textContent = `Error: ${data.message}`;
          state.history.pop(); // drop the unanswered question
          state.current = null;
        }
        setBusy(false);
      }
      showError(data.message);
      break;
  }
});

worker.addEventListener("error", (e) => {
  setStatus("The model worker failed to start", "error");
  showError(e.message || "Could not load transformers.js. Is cdn.jsdelivr.net reachable?");
  $("load").disabled = false;
});

function showError(message) {
  const el = document.createElement("p");
  el.className = "error-note";
  el.textContent = message;
  $("progress").append(el);
}

// ---------------------------------------------------------------- wire up

$("load").addEventListener("click", loadModel);
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  send($("prompt").value);
});
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send($("prompt").value);
  }
});
$("stop").addEventListener("click", () => worker.postMessage({ type: "interrupt" }));
$("clear").addEventListener("click", () => {
  if (state.busy) return;
  state.history = [];
  $("messages").innerHTML = `<p class="empty muted" id="empty">${state.ready ? "Ask a question to begin." : "Load a model, then ask it anything."}</p>`;
  $("suggestions").hidden = false;
  $("stats").textContent = "";
});
for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => send(chip.textContent));
}

initPresets();
setBusy(false);
if (!navigator.gpu) {
  setStatus("WebGPU is not available in this browser", "error");
  $("load").disabled = true;
} else {
  setStatus("Pick a model and load it", "idle");
}
