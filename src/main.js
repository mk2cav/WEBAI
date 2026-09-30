import { getKernel } from "../vendor/huggingface-kernels/index.js";
import { moeReference, compare, rng, randn, route } from "./moe-reference.js";

const REPO = "webgpu-kernels/com.microsoft.MoE";
const HUB = `https://huggingface.co/kernels/${REPO}/resolve/v1`;

// ONNX Runtime's input order for com.microsoft.MoE.
const ORT_INPUTS = [
  "input",
  "router_probs",
  "fc1_experts_weights",
  "fc1_experts_bias",
  "fc2_experts_weights",
  "fc2_experts_bias",
  "fc3_experts_weights",
  "fc3_experts_bias",
];

const $ = (id) => document.getElementById(id);
const state = { kernel: null, manifest: null, names: null, dtype: "float32", layout: null, last: null };

// ---------------------------------------------------------------- logging

function log(msg, kind = "info") {
  const line = document.createElement("div");
  line.className = `log-${kind}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  $("log").prepend(line);
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.dataset.kind = kind;
}

// ---------------------------------------------------------------- float16 helpers

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

function toHalfBits(v) {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >>> (1 - exp);
    return sign | ((mant + 0x1000) >>> 13);
  }
  return (sign | (exp << 10) | (mant >>> 13)) + ((mant >>> 12) & 1);
}

function fromHalfBits(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function toDevice(arr) {
  if (state.dtype === "float32") return arr;
  if (typeof Float16Array !== "undefined") return Float16Array.from(arr);
  return Uint16Array.from(arr, toHalfBits); // raw binary16 words
}

function toFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) return Float32Array.from(data, fromHalfBits);
  return Float32Array.from(data);
}

// ---------------------------------------------------------------- manifest

async function fetchManifest() {
  try {
    const res = await fetch(`${HUB}/build/webgpu/manifest.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    log(`Could not read manifest.json (${err.message}); falling back to ONNX Runtime names.`, "warn");
    return null;
  }
}

function resolveNames(manifest) {
  const names = {};
  const declared = manifest?.inputs ?? {};
  for (const ort of ORT_INPUTS) {
    const hit = Object.entries(declared).find(([name, spec]) => name === ort || spec?.onnx === ort);
    names[ort] = hit ? hit[0] : manifest ? null : ort;
  }
  const outs = Object.keys(manifest?.outputs ?? {});
  names.output =
    outs.find((n) => n === "output" || manifest.outputs[n]?.onnx === "output") ?? outs[0] ?? "output";
  return names;
}

function pickDtype(manifest) {
  const spec = manifest?.inputs?.[state.names?.input];
  if (!spec?.dtype) return "float32";
  const allowed = manifest.typeConstraints?.[spec.dtype] ?? [spec.dtype];
  return allowed.includes("float32") ? "float32" : allowed.includes("float16") ? "float16" : allowed[0];
}

function renderManifest(manifest) {
  const box = $("manifest");
  if (!manifest) {
    box.innerHTML = `<p class="muted">Manifest unavailable, so the demo uses ONNX Runtime's input names.</p>`;
    return;
  }
  const row = (name, spec) => `
    <tr>
      <td><code>${name}</code></td>
      <td>${spec.onnx ? `<code>${spec.onnx}</code>` : ""}</td>
      <td>${spec.dtype ?? ""}</td>
      <td>${spec.rank ?? ""}</td>
      <td>${spec.optional ? "optional" : ""}</td>
    </tr>`;
  const table = (title, obj) => `
    <h4>${title}</h4>
    <table><thead><tr><th>Name</th><th>ONNX</th><th>Type</th><th>Rank</th><th></th></tr></thead>
    <tbody>${Object.entries(obj ?? {}).map(([n, s]) => row(n, s ?? {})).join("")}</tbody></table>`;
  const attrs = Object.entries(manifest.attributes ?? {})
    .map(([k, v]) => `<li><code>${k}</code> = <code>${JSON.stringify(v)}</code></li>`)
    .join("");
  box.innerHTML = `
    ${manifest.description ? `<p>${escapeHtml(manifest.description)}</p>` : ""}
    ${table("Inputs", manifest.inputs)}
    ${table("Outputs", manifest.outputs)}
    ${attrs ? `<h4>Attributes (defaults)</h4><ul class="attrs">${attrs}</ul>` : ""}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// ---------------------------------------------------------------- load

async function load() {
  if (!navigator.gpu) {
    setStatus("WebGPU is not available in this browser", "error");
    log("navigator.gpu is missing. Use a recent Chrome, Edge or Safari with WebGPU enabled.", "error");
    return;
  }
  $("load").disabled = true;
  setStatus("Loading kernel from the Hub…", "busy");
  try {
    const t0 = performance.now();
    const [kernel, manifest] = await Promise.all([getKernel(REPO, { version: 1 }), fetchManifest()]);
    state.kernel = kernel;
    state.manifest = manifest;
    state.names = resolveNames(manifest);
    state.dtype = pickDtype(manifest);
    renderManifest(manifest);
    log(`Loaded ${kernel.opId} in ${(performance.now() - t0).toFixed(0)} ms (tensors: ${state.dtype}).`, "ok");
    setStatus(`Loaded ${kernel.opId}`, "ok");
    $("run").disabled = false;
    $("bench").disabled = false;
  } catch (err) {
    setStatus("Failed to load kernel", "error");
    const hint = err instanceof TypeError ? " (network error: can this page reach huggingface.co?)" : "";
    log(err.message + hint, "error");
    $("load").disabled = false;
  }
}

// ---------------------------------------------------------------- problem setup

function readConfig() {
  const num = (id) => Number($(id).value);
  const cfg = {
    rows: num("rows"),
    hidden: num("hidden"),
    inter: num("inter"),
    numExperts: num("experts"),
    k: num("k"),
    activation: $("activation").value,
    normalize: $("normalize").checked,
    gated: $("gated").checked,
    bias: $("bias").checked,
    seed: num("seed"),
  };
  if (cfg.k > cfg.numExperts) throw new Error("top-k cannot exceed the number of experts");
  return cfg;
}

function makeProblem(cfg) {
  const rand = rng(cfg.seed);
  const { rows, hidden, inter, numExperts: E } = cfg;
  return {
    ...cfg,
    input: randn(rows * hidden, rand),
    routerLogits: randn(rows * E, rand, 1.5),
    fc1W: randn(E * hidden * inter, rand, 1 / Math.sqrt(hidden)),
    fc2W: randn(E * inter * hidden, rand, 1 / Math.sqrt(inter)),
    fc3W: cfg.gated ? randn(E * hidden * inter, rand, 1 / Math.sqrt(hidden)) : null,
    fc1B: cfg.bias ? randn(E * inter, rand, 0.1) : null,
    fc2B: cfg.bias ? randn(E * hidden, rand, 0.1) : null,
    fc3B: cfg.bias && cfg.gated ? randn(E * inter, rand, 0.1) : null,
  };
}

// layout "oi": fc1 [E, inter, hidden] (ORT ≥ 1.22); "io": fc1 [E, hidden, inter].
function buildCall(p, layout) {
  const n = state.names;
  const { rows, hidden, inter, numExperts: E } = p;
  const t = (data, shape) => ({ data: toDevice(data), shape });
  const inputs = {};
  const put = (ort, tensor) => {
    if (!tensor) return;
    if (!n[ort]) throw new Error(`This kernel build does not declare the ONNX input "${ort}".`);
    inputs[n[ort]] = tensor;
  };
  const up = layout === "oi" ? [E, inter, hidden] : [E, hidden, inter];
  const down = layout === "oi" ? [E, hidden, inter] : [E, inter, hidden];
  put("input", t(p.input, [rows, hidden]));
  put("router_probs", t(p.routerLogits, [rows, E]));
  put("fc1_experts_weights", t(p.fc1W, up));
  put("fc1_experts_bias", p.fc1B && t(p.fc1B, [E, inter]));
  put("fc2_experts_weights", t(p.fc2W, down));
  put("fc2_experts_bias", p.fc2B && t(p.fc2B, [E, hidden]));
  put("fc3_experts_weights", p.fc3W && t(p.fc3W, up));
  put("fc3_experts_bias", p.fc3B && t(p.fc3B, [E, inter]));

  const wanted = {
    k: p.k,
    activation_type: p.activation,
    normalize_routing_weights: p.normalize ? 1 : 0,
  };
  const declared = state.manifest?.attributes;
  const attrs = declared
    ? Object.fromEntries(Object.entries(wanted).filter(([key]) => key in declared))
    : wanted;
  return { inputs, attrs };
}

// ---------------------------------------------------------------- run

async function run() {
  $("run").disabled = true;
  setStatus("Running on the GPU…", "busy");
  try {
    const cfg = readConfig();
    const p = makeProblem(cfg);
    const { result, gpuMs, layout } = await callWithLayout(p);
    const out = result[state.names.output] ?? Object.values(result)[0];
    const gpu = toFloat32(out.data);

    const t1 = performance.now();
    const ref = moeReference({ ...p, layout });
    const cpuMs = performance.now() - t1;
    const best = { layout, ...ref, err: compare(gpu, ref.output) };
    const tol = state.dtype === "float16" ? 5e-2 : 2e-3;
    const pass = best.err.maxRel < tol;

    state.last = { cfg, routing: best.routing, gpu, ref: best.output };
    renderResults({ cfg, out, gpuMs, cpuMs, best, pass, tol });
    renderRouting(cfg, best.routing);
    log(
      `${cfg.rows} tokens × ${cfg.hidden} hidden, ${cfg.numExperts} experts, top-${cfg.k}: ` +
        `GPU ${gpuMs.toFixed(1)} ms, max |Δ| ${best.err.maxAbs.toExponential(2)} (${pass ? "match" : "MISMATCH"})`,
      pass ? "ok" : "warn",
    );
    setStatus(pass ? "Output matches the CPU reference" : "Output differs from the CPU reference", pass ? "ok" : "warn");
  } catch (err) {
    setStatus("Run failed", "error");
    log(err.message, "error");
    console.error(err);
  } finally {
    $("run").disabled = false;
  }
}

// Weight layout changed across ONNX Runtime releases. Try the current one first and
// fall back to the older one if the kernel rejects the shapes; remember what worked.
async function callWithLayout(p) {
  const order = state.layout ? [state.layout] : ["oi", "io"];
  let lastErr;
  for (const layout of order) {
    try {
      const { inputs, attrs } = buildCall(p, layout);
      const t0 = performance.now();
      const result = await state.kernel(inputs, { attrs });
      const gpuMs = performance.now() - t0;
      if (state.layout !== layout) log(`Kernel accepted weight layout ${layoutName(layout)}.`);
      state.layout = layout;
      return { result, gpuMs, layout };
    } catch (err) {
      lastErr = err;
      if (order.length > 1) log(`Layout ${layoutName(layout)} rejected: ${err.message}`, "warn");
    }
  }
  throw lastErr;
}

const layoutName = (l) => (l === "oi" ? "[E, out, in]" : "[E, in, out]");

async function bench() {
  $("bench").disabled = true;
  setStatus("Benchmarking…", "busy");
  try {
    const cfg = readConfig();
    const p = makeProblem(cfg);
    await callWithLayout(p); // warm-up (pipeline compile) and layout negotiation
    const { inputs, attrs } = buildCall(p, state.layout);
    const iters = 20;
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) await state.kernel(inputs, { attrs });
    const ms = (performance.now() - t0) / iters;
    const flops = 2 * cfg.rows * cfg.k * cfg.hidden * cfg.inter * (cfg.gated ? 3 : 2);
    $("bench-out").textContent =
      `${ms.toFixed(2)} ms per call (upload, compute and readback, mean of ${iters}) · ` +
      `${(flops / ms / 1e6).toFixed(2)} GFLOP/s of expert FFN work`;
    log(`Benchmark: ${ms.toFixed(2)} ms/call`, "ok");
    setStatus("Benchmark done", "ok");
  } catch (err) {
    setStatus("Benchmark failed", "error");
    log(err.message, "error");
  } finally {
    $("bench").disabled = false;
  }
}

// ---------------------------------------------------------------- rendering

function renderResults({ cfg, out, gpuMs, cpuMs, best, pass, tol }) {
  $("results").hidden = false;
  $("m-shape").textContent = `[${out.shape.join(", ")}]`;
  $("m-dtype").textContent = out.dtype;
  $("m-gpu").textContent = `${gpuMs.toFixed(1)} ms`;
  $("m-cpu").textContent = `${cpuMs.toFixed(1)} ms`;
  $("m-err").textContent = best.err.maxAbs.toExponential(2);
  $("m-rel").textContent = best.err.maxRel.toExponential(2);
  $("m-layout").textContent = layoutName(best.layout);
  const verdict = $("verdict");
  verdict.textContent = pass ? `✓ matches (rel. tol ${tol})` : `✗ differs (rel. tol ${tol})`;
  verdict.dataset.kind = pass ? "ok" : "warn";

  const n = Math.min(cfg.hidden, 8);
  const rows = Math.min(cfg.rows, 4);
  let html = "<thead><tr><th>token</th><th>source</th>";
  for (let i = 0; i < n; i++) html += `<th>h${i}</th>`;
  html += "</tr></thead><tbody>";
  for (let r = 0; r < rows; r++) {
    for (const [label, arr] of [["GPU", state.last.gpu], ["CPU", state.last.ref]]) {
      html += `<tr><td>${label === "GPU" ? r : ""}</td><td>${label}</td>`;
      for (let i = 0; i < n; i++) html += `<td>${arr[r * cfg.hidden + i].toFixed(4)}</td>`;
      html += "</tr>";
    }
  }
  $("sample").innerHTML = html + "</tbody>";
}

function renderRouting(cfg, routing) {
  const grid = $("heatmap");
  grid.style.gridTemplateColumns = `3.5rem repeat(${cfg.numExperts}, minmax(1.4rem, 1fr))`;
  const showRows = Math.min(cfg.rows, 32);
  let html = `<div class="hm-head"></div>`;
  for (let e = 0; e < cfg.numExperts; e++) html += `<div class="hm-head">E${e}</div>`;
  for (let r = 0; r < showRows; r++) {
    html += `<div class="hm-label">tok ${r}</div>`;
    const { probs, experts, weights } = routing[r];
    for (let e = 0; e < cfg.numExperts; e++) {
      const slot = experts.indexOf(e);
      const title = `token ${r} → expert ${e}: p=${probs[e].toFixed(3)}` + (slot >= 0 ? `, weight ${weights[slot].toFixed(3)}` : "");
      html += `<div class="hm-cell${slot >= 0 ? " chosen" : ""}" style="--p:${probs[e].toFixed(3)}" title="${title}"></div>`;
    }
  }
  grid.innerHTML = html;
  $("heatmap-note").textContent =
    cfg.rows > showRows ? `Showing the first ${showRows} of ${cfg.rows} tokens.` : "";

  const load = new Array(cfg.numExperts).fill(0);
  for (const r of routing) for (const e of r.experts) load[e]++;
  const max = Math.max(...load, 1);
  $("load-bars").innerHTML = load
    .map(
      (c, e) => `
      <div class="bar-row">
        <span class="bar-label">E${e}</span>
        <span class="bar-track"><span class="bar" style="width:${(100 * c) / max}%"></span></span>
        <span class="bar-val">${c}</span>
      </div>`,
    )
    .join("");
}

// Live routing preview while editing the config (CPU only, cheap).
function preview() {
  try {
    const cfg = readConfig();
    const rand = rng(cfg.seed);
    randn(cfg.rows * cfg.hidden, rand); // keep the stream aligned with makeProblem
    const logits = randn(cfg.rows * cfg.numExperts, rand, 1.5);
    renderRouting(cfg, route(logits, cfg.rows, cfg.numExperts, cfg.k, cfg.normalize));
  } catch {
    /* invalid config while typing */
  }
}

// ---------------------------------------------------------------- wire up

$("load").addEventListener("click", load);
$("run").addEventListener("click", run);
$("bench").addEventListener("click", bench);
for (const id of ["rows", "hidden", "inter", "experts", "k", "normalize", "seed"]) {
  $(id).addEventListener("input", preview);
}
$("reseed").addEventListener("click", () => {
  $("seed").value = Math.floor(Math.random() * 1e6);
  preview();
});
if (!navigator.gpu) {
  setStatus("WebGPU is not available in this browser", "error");
} else {
  setStatus("Ready: load the kernel to begin", "idle");
}
preview();
