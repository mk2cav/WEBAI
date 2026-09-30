import { test } from "node:test";
import assert from "node:assert/strict";
import { moeReference, route, rng, randn, erf } from "../src/moe-reference.js";

const transpose = (w, E, a, b) => {
  const t = new Float32Array(w.length);
  for (let e = 0; e < E; e++)
    for (let i = 0; i < a; i++)
      for (let j = 0; j < b; j++) t[e * a * b + j * a + i] = w[e * a * b + i * b + j];
  return t;
};

test("erf matches known values", () => {
  assert.ok(Math.abs(erf(0.5) - 0.5204998778) < 1e-6);
  assert.ok(Math.abs(erf(-1) + 0.8427007929) < 1e-6);
});

test("routing picks top-k and renormalises", () => {
  const logits = Float32Array.from([0, 2, 1, -1]);
  const [r] = route(logits, 1, 4, 2, true);
  assert.deepEqual(r.experts, [1, 2]);
  assert.ok(Math.abs(r.weights[0] + r.weights[1] - 1) < 1e-12);
});

test("single expert, identity activation is a plain two-layer linear map", () => {
  // hidden=2, inter=2, W1 = I, W2 = 2I, no bias -> y = 2x
  const p = {
    rows: 1, hidden: 2, inter: 2, numExperts: 1, k: 1, activation: "identity", normalize: false, layout: "io",
    input: Float32Array.from([3, -4]), routerLogits: Float32Array.from([0]),
    fc1W: Float32Array.from([1, 0, 0, 1]), fc2W: Float32Array.from([2, 0, 0, 2]),
  };
  assert.deepEqual([...moeReference(p).output], [6, -8]);
});

test("the two weight layouts agree once transposed", () => {
  const rand = rng(7);
  const [rows, hidden, inter, E] = [5, 6, 10, 4];
  const base = {
    rows, hidden, inter, numExperts: E, k: 2, activation: "gelu", normalize: true,
    input: randn(rows * hidden, rand), routerLogits: randn(rows * E, rand),
    fc1B: randn(E * inter, rand), fc2B: randn(E * hidden, rand), fc3B: randn(E * inter, rand),
  };
  const fc1 = randn(E * hidden * inter, rand), fc2 = randn(E * inter * hidden, rand), fc3 = randn(E * hidden * inter, rand);
  const io = moeReference({ ...base, layout: "io", fc1W: fc1, fc2W: fc2, fc3W: fc3 }).output;
  const oi = moeReference({
    ...base, layout: "oi",
    fc1W: transpose(fc1, E, hidden, inter), fc2W: transpose(fc2, E, inter, hidden), fc3W: transpose(fc3, E, hidden, inter),
  }).output;
  for (let i = 0; i < io.length; i++) assert.ok(Math.abs(io[i] - oi[i]) < 1e-5);
});
