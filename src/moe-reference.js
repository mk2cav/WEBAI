// CPU reference for the ONNX Runtime contrib op `com.microsoft.MoE`.
//
//   probs   = softmax(router_probs[row])              (router_probs are logits)
//   experts = top-k(probs)                            (ties -> lower expert index)
//   weights = probs[experts]  (renormalised to sum 1 if normalize_routing_weights)
//   out     = Σ weights[e] · FFN_e(input[row])
//   FFN_e(x) = (act(x·W1ₑ + b1ₑ) [⊙ (x·W3ₑ + b3ₑ)]) · W2ₑ + b2ₑ
//
// Weight layout differs between ORT releases, so both are supported:
//   "io" : fc1 [E, hidden, inter],  fc2 [E, inter, hidden]   (x · W)
//   "oi" : fc1 [E, inter, hidden],  fc2 [E, hidden, inter]   (x · Wᵀ)

export const ACTIVATIONS = {
  identity: (v) => v,
  relu: (v) => (v > 0 ? v : 0),
  gelu: (v) => 0.5 * v * (1 + erf(v / Math.SQRT2)),
  silu: (v) => v / (1 + Math.exp(-v)),
};

// Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
export function erf(x) {
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-a * a);
  return s * y;
}

export function softmax(row) {
  let max = -Infinity;
  for (const v of row) max = Math.max(max, v);
  const out = new Float64Array(row.length);
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += out[i] = Math.exp(row[i] - max);
  for (let i = 0; i < row.length; i++) out[i] /= sum;
  return out;
}

/** Per-row routing: softmax probabilities, chosen experts and their mixing weights. */
export function route(routerLogits, rows, numExperts, k, normalize) {
  const result = [];
  for (let r = 0; r < rows; r++) {
    const probs = softmax(routerLogits.subarray(r * numExperts, (r + 1) * numExperts));
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a] || a - b);
    const experts = order.slice(0, k);
    let weights = experts.map((e) => probs[e]);
    if (normalize) {
      const s = weights.reduce((a, b) => a + b, 0);
      weights = weights.map((w) => w / s);
    }
    result.push({ probs, experts, weights });
  }
  return result;
}

/**
 * @param {object} p
 * @param {Float32Array} p.input        [rows, hidden]
 * @param {Float32Array} p.routerLogits [rows, experts]
 * @param {Float32Array} p.fc1W, p.fc2W, p.fc3W?  expert weights (see layout above)
 * @param {Float32Array} p.fc1B?, p.fc2B?, p.fc3B?  [E, inter] / [E, hidden] / [E, inter]
 */
export function moeReference(p) {
  const { rows, hidden, inter, numExperts, k, activation, normalize, layout } = p;
  const act = ACTIVATIONS[activation];
  if (!act) throw new Error(`unsupported activation ${activation}`);
  const out = new Float32Array(rows * hidden);
  const routing = route(p.routerLogits, rows, numExperts, k, normalize);
  const h = new Float64Array(inter);

  // Index helpers for the two layouts.
  const w1 = layout === "io"
    ? (e, i, j) => e * hidden * inter + i * inter + j   // [E, hidden, inter]
    : (e, i, j) => e * inter * hidden + j * hidden + i; // [E, inter, hidden]
  const w2 = layout === "io"
    ? (e, j, i) => e * inter * hidden + j * hidden + i  // [E, inter, hidden]
    : (e, j, i) => e * hidden * inter + i * inter + j;  // [E, hidden, inter]

  for (let r = 0; r < rows; r++) {
    const x = p.input.subarray(r * hidden, (r + 1) * hidden);
    const { experts, weights } = routing[r];
    for (let t = 0; t < experts.length; t++) {
      const e = experts[t];
      for (let j = 0; j < inter; j++) {
        let a = p.fc1B ? p.fc1B[e * inter + j] : 0;
        for (let i = 0; i < hidden; i++) a += x[i] * p.fc1W[w1(e, i, j)];
        let v = act(a);
        if (p.fc3W) {
          let g = p.fc3B ? p.fc3B[e * inter + j] : 0;
          for (let i = 0; i < hidden; i++) g += x[i] * p.fc3W[w1(e, i, j)];
          v *= g;
        }
        h[j] = v;
      }
      for (let i = 0; i < hidden; i++) {
        let y = p.fc2B ? p.fc2B[e * hidden + i] : 0;
        for (let j = 0; j < inter; j++) y += h[j] * p.fc2W[w2(e, j, i)];
        out[r * hidden + i] += weights[t] * y;
      }
    }
  }
  return { output: out, routing };
}

/** Max absolute / relative error between two arrays. */
export function compare(a, b) {
  let maxAbs = 0;
  let maxRef = 0;
  for (let i = 0; i < a.length; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    maxRef = Math.max(maxRef, Math.abs(b[i]));
  }
  return { maxAbs, maxRel: maxAbs / (maxRef || 1) };
}

/** Deterministic PRNG (mulberry32) so runs are reproducible from a seed. */
export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(n, rand, scale = 1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const m = Math.sqrt(-2 * Math.log(u)) * scale;
    a[i] = m * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) a[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return a;
}
