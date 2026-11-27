// Interactive mode for opencode. UNTESTED: the tested setup is autonomous, which is this plugin with the mode off.
//
// Two modes, per session:
//   autonomous   changes nothing the model sees: every request is exactly what it would be without this plugin.
//   interactive  appends a short <interaction> block to the end of opencode's one system message (the main agent's
//                steps only: sub-agent sessions, the title writer and the summarizer are left alone; their requests
//                carry no environment lines). The block says the user is at the keyboard and when to ask them.
//
// /interactive and /autonomous switch the current session's mode (this plugin adds both commands to opencode's config,
// unless you have your own commands of those names). The command's message is replaced by one line saying the mode
// changed, marked synthetic (not typed by the user) so that a task driver does not take it for a new request; the model
// sees that line and replies (opencode's TUI does not show synthetic text: a toast says which mode is on). The switch
// lasts for this session while opencode runs; a new session starts in the default.
//
// Options (plugin entry [path, {default, dir}]): default = "interactive" or "autonomous" (anything else: autonomous), the
// mode a session starts in; dir = where .events.jsonl goes (loaded / mode / shown, one JSON object per line).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// opencode loads every export of a plugin file as a plugin, so only the plugin itself is exported
const BLOCK = [
  "<interaction>",
  "The user is at the keyboard in this session and can answer your questions.",
  "- If the request leaves a choice only the user can make (how something should behave or look, data that matters) or something in it is genuinely unclear, first take a quick look at the code so your questions are specific, then ask them all at once, early, before the long work.",
  "- If the request is a list of tasks, ask before you file the tasks or start on them.",
  "- After asking, go on with the parts that do not depend on the answers.",
  "- Ask before anything destructive or hard to undo.",
  "- Never ask whether to keep going, and never ask the user to paste or explain again something already in this conversation.",
  "- Once your questions are answered, keep going until all of the work is done.",
  "</interaction>",
].join("\n");

const COMMANDS = {
  interactive: {
    mode: "interactive",
    description: "ask me when a decision is mine (untested mode)",
    toast: "Interactive mode (untested): it asks when it needs a decision",
    note: "Mode changed to interactive: I am at the keyboard and can answer questions. Carry on with any work in progress.",
  },
  autonomous: {
    mode: "autonomous",
    description: "work without stopping to ask (the tested mode)",
    toast: "Autonomous mode: it won't stop to ask",
    note: "Mode changed to autonomous: I am leaving this to you, so ask only if you cannot continue without me. Carry on with any work in progress.",
  },
};

// opencode 1.18.34 puts this line in the main agent's system message (its environment lines); the title writer and the
// summarizer get none
const ENV = "Here is some useful information about the environment you are running in:";

export const Interactive = async ({ client } = {}, options = {}) => {
  const start = options && options.default === "interactive" ? "interactive" : "autonomous";
  const dir = String((options && options.dir) || path.join(os.homedir(), ".local", "share", "opencode", "interactive"));
  const event = (o) => {
    try {
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(path.join(dir, ".events.jsonl"), JSON.stringify({ t: Date.now(), ...o }) + "\n");
    } catch {}
  };
  event({ event: "loaded", default: start });

  const modes = new Map();      // session id -> mode set by a command
  const children = new Set();   // sub-agent sessions: never get the block
  const ours = new Set();       // the commands this plugin added (a user's own command of the same name is left alone)
  const shown = new Set();      // sessions whose block was logged since their last switch
  const modeOf = (sid) => modes.get(sid) || start;

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
      const info = e && (e.type === "session.created" || e.type === "session.updated") && e.properties && e.properties.info;
      if (info && info.id && info.parentID) children.add(info.id);
    },

    "command.execute.before": async (input, output) => {
      const name = input && input.command;
      if (!ours.has(name)) return;
      const c = COMMANDS[name], sid = input.sessionID;
      if (sid) { modes.set(sid, c.mode); shown.delete(sid); }
      // opencode sends the array it passed in (a new array here would be ignored), so it is changed in place
      const parts = output && output.parts;
      if (Array.isArray(parts)) parts.splice(0, parts.length, { type: "text", text: c.note, synthetic: true });
      event({ event: "mode", session: sid || null, mode: c.mode });
      try {
        const r = client && client.tui && client.tui.showToast && client.tui.showToast({ body: { message: c.toast, variant: "info" } });
        if (r && typeof r.catch === "function") r.catch(() => {});
      } catch {}
    },

    "experimental.chat.system.transform": async (input, output) => {
      const sid = input && input.sessionID;
      if (!sid || modeOf(sid) !== "interactive" || children.has(sid)) return;
      const sys = output && output.system;
      if (!Array.isArray(sys) || !sys.length) return;
      const last = sys.length - 1;
      if (typeof sys[last] !== "string" || !sys.some((s) => typeof s === "string" && s.includes(ENV))) return;
      if (sys[last].endsWith(BLOCK)) return;
      sys[last] = `${sys[last]}\n\n${BLOCK}`;   // into the one system message, at its end
      if (!shown.has(sid)) { shown.add(sid); event({ event: "shown", session: sid }); }
    },
  };
};
