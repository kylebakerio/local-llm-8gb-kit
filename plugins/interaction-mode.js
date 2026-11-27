// Interaction modes for opencode: autonomous (the tested mode), plan-first and interactive (both UNTESTED).
//
// The kit's system prompt (prompts/v1-2.md) has a "# Questions" section: work without asking until the work is done; where
// something is unclear, make the most sensible assumption, note it and keep going. That is autonomous, and in that mode
// this plugin leaves the system prompt exactly as it is. The other two modes replace that section's lines (its heading
// and every other section stay as they are), so the prompt never says both "don't ask" and "ask":
//   interactive  the user is at the keyboard: look first, ask early and all at once, go on with what does not depend on
//                the answers, ask before anything destructive.
//   plan-first   the user is here now but will be away: ask any questions in one round before changing anything, then
//                work without asking. The planning window ends at the first file-changing tool call (edit, write,
//                patch, or a bash command that plainly writes files), at file_tasks / add_task, when the user answers
//                a question-tool call, or at the user's first message after the model's first reply, whichever comes
//                first. From then on the session is autonomous: the section goes back to the prompt's own (the next
//                request misses the model server's prompt cache once).
//
// opencode's question tool (in the TUI's tool list; `opencode run` sessions deny it):
//   - every call is refused (it fails with a short message) in autonomous mode and after plan-first's window;
//   - a session that is autonomous at its first message, or switched by /autonomous, also has it taken out of its tool
//     list (a session permission rule, as `opencode run` adds), so its requests list the same tools as the tested ones.
//     opencode reads that rule when a prompt starts; plan-first's window often ends mid-reply, so it only refuses calls
//     (changing the tool list then would cost a second prompt-cache miss). Only a rule this plugin added is ever undone.
//
// /interactive, /plan-first and /autonomous switch the current session (this plugin adds the commands to opencode's
// config, unless you have your own of those names). The command's message is replaced by one line saying the mode
// changed, marked synthetic (not typed by the user) so that a task driver does not take it for a new request; the model
// sees that line and replies (opencode's TUI does not show synthetic text: a toast says which mode is on). A switch lasts
// for this session while opencode runs; a new session starts in the default. Sub-agent sessions are left alone.
//
// Each main session's mode is published for other plugins (a task driver) at globalThis[Symbol.for("opencode.interaction-mode")],
// a Map of session id -> "interactive" | "plan-first" | "autonomous" ("plan-first" only during its window).
//
// Options (plugin entry [path, {default, dir}]): default = "autonomous", "plan-first" or "interactive" (anything else:
// autonomous), the mode a session starts in; dir = where .events.jsonl goes (one JSON object per line).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// opencode loads every export of a plugin file as a plugin, so only the plugin itself is exported
const MODES = ["autonomous", "plan-first", "interactive"];
const HEAD = "# Questions\n";
const SECTIONS = {
  interactive: [
    "- The user is at the keyboard and can answer questions.",
    "- If the request leaves a choice only the user can make (how something should behave or look, data that matters) or something in it is genuinely unclear, first take a quick look at the code so your questions are specific, then ask them all at once, early, before the long work. With a list of tasks, ask before you file the tasks or start on them.",
    "- After asking, go on with the parts that do not depend on the answers.",
    "- Ask before anything destructive or hard to undo.",
    "- Never ask whether to keep going, and never ask the user to paste or explain again something already in this conversation.",
    "- Once your questions are answered, keep going until all of the work is done.",
  ].join("\n"),
  "plan-first": [
    "- The user is here now but will be away while you work.",
    "- Before changing anything, take a quick look at the code. If the request leaves choices only the user can make or something in it is unclear, ask all your questions in one round, then wait for the answers. With a list of tasks, ask before you file the tasks.",
    "- After that round, or if you have no questions, work without asking anything more: make the most sensible assumption, note it, and keep going. If a part cannot be done without the user, finish all the other parts first.",
  ].join("\n"),
};
const REFUSED = "Questions wait until the end: make the most sensible assumption, note it, and keep going; list it in your final message.";
const COMMANDS = {
  interactive: {
    mode: "interactive",
    description: "ask me questions as they come up (untested)",
    toast: "Interactive (untested): it asks questions as they come up",
    note: "Mode changed to interactive: I am at the keyboard and can answer questions. Carry on with any work in progress.",
  },
  "plan-first": {
    mode: "plan-first",
    description: "ask your questions up front, then work on your own (untested)",
    toast: "Plan first (untested): it asks its questions up front, then works on its own",
    note: "Mode changed to plan first: if you have questions, ask them all now, in one round; after that, work without asking. If you have none, carry on with any work in progress.",
  },
  autonomous: {
    mode: "autonomous",
    description: "work without asking; assumptions and blocking questions at the end (the tested mode)",
    toast: "Autonomous: it works without asking and lists its assumptions at the end",
    note: "Mode changed to autonomous: I may be away, so work without asking me anything until the work is done. Carry on with any work in progress.",
  },
};
// tool calls that start the work (plan-first's window ends at the first one)
const CHANGES = new Set(["edit", "write", "patch", "apply_patch", "multiedit"]);
const FILES_TASKS = new Set(["file_tasks", "add_task"]);

// opencode 1.18.34's environment lines open with these, right after its (or the replacement) prompt; only the main
// agent's requests have them (the title writer and the summarizer get none)
const ENV_FIRST = "You are powered by the model named";
const ENV = "Here is some useful information about the environment you are running in:";

// The system text with the "# Questions" section's lines replaced by `body`. Only the prompt itself (the text before the
// environment lines) is searched, so a heading in the project's AGENTS.md is never touched. No such section (opencode's
// own prompt): the section goes at the end.
function withSection(text, body) {
  let end = text.indexOf(ENV_FIRST);
  if (end < 0) end = text.indexOf(ENV);
  const region = end < 0 ? text : text.slice(0, end);
  const at = region.startsWith(HEAD) ? 0 : region.indexOf("\n" + HEAD) + 1;
  if (at === 0 && !region.startsWith(HEAD)) return `${text}\n\n${HEAD}${body}`;
  const from = at + HEAD.length;
  let to = region.indexOf("\n\n#", from);
  if (to < 0) to = region.replace(/\s+$/, "").length;
  return text.slice(0, from) + body + text.slice(to);
}

// Whether a bash command plainly writes files: an output redirect to a file, or a command that changes files (rm, mv,
// cp, mkdir, touch, sed -i, tee FILE, git commit / checkout / ..., npm install, ...). Quoted text is ignored. A command
// it cannot read this way (a script writing a file) does not count.
const WRITERS = new Set(["rm", "rmdir", "mv", "cp", "mkdir", "touch", "ln", "chmod", "chown", "truncate", "dd", "patch", "install"]);
const GIT_WRITERS = new Set(["add", "commit", "apply", "am", "checkout", "switch", "reset", "restore", "stash", "merge", "rebase", "cherry-pick",
  "rm", "mv", "clean", "pull", "revert", "init", "clone"]);
const PACKAGERS = new Set(["npm", "pnpm", "yarn", "bun", "pip", "pip3", "cargo"]);
const INSTALLS = new Set(["install", "i", "ci", "add", "remove", "rm", "uninstall", "update", "upgrade"]);
function bashWrites(cmd) {
  if (typeof cmd !== "string" || !cmd.trim()) return false;
  const bare = cmd.replace(/'[^']*'/g, "''").replace(/"(?:[^"\\]|\\.)*"/g, '""');
  for (const m of bare.matchAll(/(?:\d|&)?>>?\|?\s*([^\s&|;<>()]*)/g)) {
    if (m[1] && m[1] !== "/dev/null") return true;
  }
  for (const seg of bare.split(/&&|\|\||[;|\n]|\$\(|`/)) {
    const w = seg.trim().split(/\s+/).filter((x) => x && !/^[A-Za-z_][A-Za-z0-9_]*=/.test(x));
    let i = 0;
    while (["sudo", "env", "command", "time", "nohup"].includes(w[i])) i++;
    const c = (w[i] || "").replace(/^.*\//, ""), args = w.slice(i + 1), sub = args.find((x) => !x.startsWith("-"));
    if (WRITERS.has(c)) return true;
    if (c === "git" && GIT_WRITERS.has(sub)) return true;
    if (PACKAGERS.has(c) && INSTALLS.has(sub)) return true;
    if ((c === "sed" || c === "perl") && args.some((x) => /^-[a-zA-Z]*i/.test(x))) return true;
    if (c === "tee" && args.some((x) => !x.startsWith("-") && x !== "/dev/null")) return true;
  }
  return false;
}

const timed = (p, ms = 3000) => Promise.race([p, new Promise((_, no) => setTimeout(() => no(new Error("timed out")), ms))]);

export const InteractionMode = async ({ client } = {}, options = {}) => {
  const start = MODES.includes(options && options.default) ? options.default : "autonomous";
  const dir = String((options && options.dir) || path.join(os.homedir(), ".local", "share", "opencode", "interaction-mode"));
  const event = (o) => {
    try {
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(path.join(dir, ".events.jsonl"), JSON.stringify({ t: Date.now(), ...o }) + "\n");
    } catch {}
  };
  event({ event: "loaded", default: start });

  const KEY = Symbol.for("opencode.interaction-mode");
  const published = globalThis[KEY] instanceof Map ? globalThis[KEY] : (globalThis[KEY] = new Map());
  const sessions = new Map();   // session id -> {mode, window, typed, replied}
  const children = new Set();   // sub-agent sessions: left alone
  const ours = new Set();       // the commands this plugin added
  const shown = new Set();      // "session:mode" already logged
  const tool = new Map();       // session id -> the question tool's rule this plugin last saw or set ("allow" / "deny")
  const deniedHere = new Set(); // sessions whose question-tool "deny" this plugin added (only those are ever undone)

  const effective = (s) => (s.mode === "plan-first" && !s.window ? "autonomous" : s.mode);
  const publish = (sid) => published.set(sid, effective(sessions.get(sid)));
  function state(sid) {
    let s = sessions.get(sid);
    if (!s) { s = { mode: start, window: start === "plan-first", typed: false, replied: false }; sessions.set(sid, s); publish(sid); }
    return s;
  }
  function closeWindow(sid, by) {
    const s = state(sid);
    if (!s.window) return;
    s.window = false; publish(sid);
    event({ event: "window-closed", session: sid, by });
  }
  // the question tool in the session's tool list: out for autonomous, back (only if this plugin took it out) otherwise
  async function syncQuestionTool(sid) {
    const want = state(sid).mode === "autonomous" ? "deny" : "allow";
    if (tool.get(sid) === want) return;
    tool.set(sid, want);
    if (want === "allow" && !deniedHere.has(sid)) return;
    try {
      const got = await timed(client.session.get({ path: { id: sid } }));
      const rules = ((got && got.data && got.data.permission) || []).filter((r) => r && r.permission === "question" && r.pattern === "*");
      const last = rules.length ? rules[rules.length - 1].action : null;
      if (want === "deny" && last === "deny") return;            // already out (an `opencode run` session)
      await timed(client.session.update({ path: { id: sid }, body: { permission: [{ permission: "question", pattern: "*", action: want }] } }));
      if (want === "deny") deniedHere.add(sid); else deniedHere.delete(sid);
      event({ event: "question-tool", session: sid, action: want });
    } catch (e) {
      tool.delete(sid);                                            // tried again at the next message; calls are refused meanwhile
      event({ event: "error", where: "question-tool", session: sid, error: String((e && e.message) || e).slice(0, 200) });
    }
  }

  return {
    config: async (cfg) => {
      if (!cfg || typeof cfg !== "object") return;
      if (!cfg.command || typeof cfg.command !== "object") cfg.command = {};
      for (const [name, c] of Object.entries(COMMANDS)) {
        if (cfg.command[name]) continue;
        cfg.command[name] = { description: c.description, template: c.note };
        ours.add(name);
      }
    },

    event: async ({ event: e } = {}) => {
      const p = (e && e.properties) || {};
      if (e && (e.type === "session.created" || e.type === "session.updated") && p.info && p.info.id && p.info.parentID) children.add(p.info.id);
      const idle = e && ((e.type === "session.idle") || (e.type === "session.status" && p.status && p.status.type === "idle"));
      const s = idle && p.sessionID && sessions.get(p.sessionID);
      if (s && s.window && s.typed) s.replied = true;              // the model has answered the request in the window
    },

    "chat.message": async (input, output) => {
      const sid = input && input.sessionID;
      if (!sid || children.has(sid)) return;
      const s = state(sid);
      const typed = ((output && output.parts) || []).some((p) => p && p.type === "text" && !p.synthetic && !p.ignored && String(p.text || "").trim());
      if (typed && s.window) {
        if (s.typed && s.replied) closeWindow(sid, "answer");    // the user's first message after the model's first reply
        else { s.typed = true; s.replied = false; }
      }
      await syncQuestionTool(sid);
    },

    "command.execute.before": async (input, output) => {
      const name = input && input.command;
      if (!ours.has(name)) return;
      const c = COMMANDS[name], sid = input.sessionID;
      // opencode sends the array it passed in (a new array here would be ignored), so it is changed in place
      const parts = output && output.parts;
      if (Array.isArray(parts)) parts.splice(0, parts.length, { type: "text", text: c.note, synthetic: true });
      if (sid) {
        Object.assign(state(sid), { mode: c.mode, window: c.mode === "plan-first", typed: false, replied: false });
        publish(sid);
        await syncQuestionTool(sid);
      }
      event({ event: "mode", session: sid || null, mode: c.mode });
      try {
        const r = client && client.tui && client.tui.showToast && client.tui.showToast({ body: { message: c.toast, variant: "info" } });
        if (r && typeof r.catch === "function") r.catch(() => {});
      } catch {}
    },

    "tool.execute.before": async (input, output) => {
      const sid = input && input.sessionID, name = input && input.tool;
      if (!sid || children.has(sid)) return;
      const s = state(sid);
      if (s.window && (CHANGES.has(name) || FILES_TASKS.has(name) || (name === "bash" && bashWrites(output && output.args && output.args.command)))) closeWindow(sid, name);
      if (name === "question" && effective(s) === "autonomous") {
        event({ event: "question-refused", session: sid });
        throw new Error(REFUSED);
      }
    },

    "tool.execute.after": async (input) => {
      const sid = input && input.sessionID;
      if (!sid || children.has(sid) || !input || input.tool !== "question") return;
      if (state(sid).window) closeWindow(sid, "question answered");
    },

    "experimental.chat.system.transform": async (input, output) => {
      const sid = input && input.sessionID;
      if (!sid || children.has(sid)) return;
      const sys = output && output.system;
      if (!Array.isArray(sys)) return;
      const mode = effective(state(sid));
      if (mode === "autonomous") return;
      const k = sys.findIndex((x) => typeof x === "string" && x.includes(ENV));
      if (k < 0) return;
      sys[k] = withSection(sys[k], SECTIONS[mode]);
      if (!shown.has(`${sid}:${mode}`)) { shown.add(`${sid}:${mode}`); event({ event: "shown", session: sid, mode }); }
    },
  };
};
