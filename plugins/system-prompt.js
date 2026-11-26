// System prompt for opencode.
//
// opencode puts one system message at the top of every request: its own general instructions first -- the prompt it
// picks by model id ("You are opencode, an interactive CLI tool ..." for most models, other wordings for gpt, claude,
// gemini, kimi, ... ids) -- then the model / environment lines, the project's instructions (AGENTS.md) and the skills
// list. This plugin replaces only the first part, opencode's own instructions, with the text of a file. Everything after
// it stays exactly as opencode wrote it, including the blank line before the environment lines. Agents that bring their
// own prompt (the title writer, the summarizer, explore) are left alone; tools, messages and what is stored are not
// touched.
//
// With the continueText option it also replaces the turn opencode adds by itself after it summarizes a long session
// ("Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.") with that text,
// in the model's view only, the same way the session-history plugin's "no-stop-and-ask" part does.
//
// Options (plugin entry [path, {file, dir, continueText}]): file = the text to use (read once, used as it is); dir = where
// .events.jsonl goes (loaded / replaced / continue-replaced / errors, one JSON object per line); continueText = optional.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";

// how opencode 1.18.34's own prompts begin (SystemPrompt.provider: the default, trinity, gpt-4 / o1 / o3, gemini-, gpt,
// gpt-6, codex, claude, kimi and muse prompts); each is always the start of the system message
const HEADS = ["You are opencode, ", "You are OpenCode, ", "You are an AI agent powered by OpenCode, "];
// where opencode's own instructions end: the first of these that follows them (the model / environment lines come first)
const ENDS = ["You are powered by the model named", "Here is some useful information about the environment", "<env>", "Instructions from:", "Skills provide specialized instructions", "<available_skills>", "Project references provide", "<available_references>", "<mcp_instructions>"];

// the automatic turn after a summary (opencode 1.18.34; the same text and matching as session-history.js)
const STOP_ASK = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.";

// null: not opencode's own prompt; undefined: opencode's prompt with no end found; else the new text
function replaceOwn(text, prompt) {
  if (typeof text !== "string") return null;
  const head = HEADS.find((h) => text.startsWith(h));
  if (!head) return null;
  const ends = ENDS.map((m) => text.indexOf(m, head.length)).filter((j) => j > 0);
  if (!ends.length) return undefined;
  const j = Math.min(...ends);
  const own = text.slice(0, j);
  const gap = own.slice(own.replace(/\s+$/, "").length);   // the prompt's last newline + the joining one, kept as they are
  return { text: prompt + gap + text.slice(j), head, own: own.length };   // own: the characters replaced
}

export const SystemPrompt = async ({ directory }, options = {}) => {
  const dir = String((options && options.dir) || path.join(os.homedir(), ".local", "share", "opencode", "system-prompt"));
  const event = (o) => {
    try {
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(path.join(dir, ".events.jsonl"), JSON.stringify({ t: Date.now(), ...o }) + "\n");
    } catch {}
  };
  let prompt = null;
  try { prompt = fs.readFileSync(String((options && options.file) || ""), "utf8"); }
  catch (e) { event({ event: "error", where: "load", error: String((e && e.message) || e).slice(0, 300) }); return {}; }
  if (!prompt.trim()) { event({ event: "error", where: "load", error: "the prompt file is empty" }); return {}; }
  const sha256 = crypto.createHash("sha256").update(prompt, "utf8").digest("hex");
  const cont = typeof (options && options.continueText) === "string" && options.continueText.trim() ? options.continueText : null;
  event({ event: "loaded", directory, sha256, chars: prompt.length, continue: !!cont });

  const seen = new Set();   // sessions (and continue turns) already logged in this process
  let misses = 0;
  const hooks = {
    "experimental.chat.system.transform": async (input, output) => {
      if (!output || !Array.isArray(output.system)) return;
      for (let k = 0; k < output.system.length; k++) {
        const r = replaceOwn(output.system[k], prompt);
        if (r === null) continue;
        if (r === undefined) {
          if (misses++ === 0) event({ event: "error", where: "split", error: "opencode's own prompt is there but its end was not found; left as it is" });
          continue;
        }
        const before = output.system[k].length;
        output.system[k] = r.text;
        const sid = input && input.sessionID;
        if (!seen.has(sid)) {
          seen.add(sid);
          const m = (input && input.model) || {};
          event({ event: "replaced", session: sid, model: (m.api && m.api.id) || m.id || null, provider: m.providerID || null,
                  head: r.head, own_chars: r.own, chars_before: before, chars_after: r.text.length, sha256 });
        }
        break;
      }
    },
  };
  if (cont) hooks["experimental.chat.messages.transform"] = async (_input, output) => {
    const msgs = output && output.messages;
    if (!Array.isArray(msgs)) return;
    for (let i = 0; i < msgs.length; i++) {
      const m = msgs[i];
      if (!m || !m.info || m.info.role !== "user" || !Array.isArray(m.parts)) continue;
      const k = m.parts.findIndex((p) => p.type === "text" && p.synthetic && typeof p.text === "string" && p.text.endsWith(STOP_ASK));
      if (k < 0) continue;
      const ps = [...m.parts];
      ps[k] = { ...ps[k], text: ps[k].text.slice(0, -STOP_ASK.length) + cont };
      msgs[i] = { ...m, parts: ps };
      const key = `${m.info.sessionID}:${m.info.id}:continue`;
      if (!seen.has(key)) { seen.add(key); event({ event: "continue-replaced", session: m.info.sessionID, message: m.info.id }); }
    }
  };
  return hooks;
};
