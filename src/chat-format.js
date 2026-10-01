// Split a model's raw streamed output into hidden reasoning and the visible answer.
//
// Handles the two common conventions:
//   • <think> … </think> answer                         (Qwen3, DeepSeek-R1 style)
//   • <|channel|>analysis<|message|> … <|end|>
//     <|start|>assistant<|channel|>final<|message|> answer (gpt-oss "harmony" format)
// Any other special tokens (<|im_end|>, <|return|>, </s>, …) are stripped.

const SPECIAL = /<\|[^|<>\s]*\|>|<\/?s>|<end_of_turn>|<start_of_turn>/g;
const clean = (s) => s.replace(SPECIAL, "").trim();

export function parseOutput(raw) {
  // gpt-oss harmony channels
  const finalMarker = /<\|channel\|>final<\|message\|>/;
  const analysisMarker = /<\|channel\|>analysis<\|message\|>/;
  if (analysisMarker.test(raw) || finalMarker.test(raw)) {
    const finalMatch = raw.match(finalMarker);
    const head = finalMatch ? raw.slice(0, finalMatch.index) : raw;
    const answer = finalMatch ? raw.slice(finalMatch.index + finalMatch[0].length) : "";
    const a = head.match(analysisMarker);
    const reasoning = a ? head.slice(a.index + a[0].length).split("<|end|>")[0] : "";
    return { reasoning: clean(reasoning), answer: clean(answer), thinking: !finalMatch };
  }

  // <think> … </think>
  const open = raw.indexOf("<think>");
  const close = raw.indexOf("</think>");
  if (open !== -1 || close !== -1) {
    const start = open === -1 ? 0 : open + "<think>".length;
    if (close === -1) return { reasoning: clean(raw.slice(start)), answer: "", thinking: true };
    return {
      reasoning: clean(raw.slice(start, close)),
      answer: clean(raw.slice(close + "</think>".length)),
      thinking: false,
    };
  }

  return { reasoning: "", answer: clean(raw), thinking: false };
}
