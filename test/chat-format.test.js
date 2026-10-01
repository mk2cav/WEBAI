import { test } from "node:test";
import assert from "node:assert/strict";
import { parseOutput } from "../src/chat-format.js";

test("plain output strips end-of-turn tokens", () => {
  assert.deepEqual(parseOutput("Paris is the capital.<|im_end|>"), {
    reasoning: "", answer: "Paris is the capital.", thinking: false,
  });
});

test("<think> blocks become reasoning, streaming and complete", () => {
  assert.deepEqual(parseOutput("<think>\nThe user asks"), {
    reasoning: "The user asks", answer: "", thinking: true,
  });
  assert.deepEqual(parseOutput("<think>hmm</think>\n\n42<|im_end|>"), {
    reasoning: "hmm", answer: "42", thinking: false,
  });
});

test("gpt-oss harmony channels split into reasoning and final answer", () => {
  const partial = "<|channel|>analysis<|message|>We need to add.";
  assert.deepEqual(parseOutput(partial), { reasoning: "We need to add.", answer: "", thinking: true });

  const full =
    "<|channel|>analysis<|message|>We need to add.<|end|>" +
    "<|start|>assistant<|channel|>final<|message|>2 + 2 = 4<|return|>";
  assert.deepEqual(parseOutput(full), { reasoning: "We need to add.", answer: "2 + 2 = 4", thinking: false });
});
