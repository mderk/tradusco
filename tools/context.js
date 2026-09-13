#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { parseObjects, stringify } = require("./lib/csv");

function args(argv) {
  const out = { _: [] };
  for (let i = 2; i < argv.length; i++) {
    const value = argv[i];
    if (!value.startsWith("--")) out._.push(value);
    else {
      const key = value.slice(2);
      const next = argv[i + 1];
      out[key] = !next || next.startsWith("--") ? true : next;
      if (out[key] !== true) i++;
    }
  }
  return out;
}

function readJson(file, fallback) {
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : fallback;
}

function writeJson(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, file);
}

function writeCsv(file, head, rows) {
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, `${stringify([head, ...rows.map((row) => head.map((key) => row[key]))])}\n`, "utf8");
  fs.renameSync(temporary, file);
}

function load(options) {
  const configFile = path.resolve(String(options.config || ".tradusco/config.json"));
  const root = path.dirname(configFile);
  const config = readJson(configFile, null);
  if (!config) throw new Error(`config not found: ${configFile}`);
  const projectDir = path.resolve(root, config.projectDir || "project");
  const projectConfig = readJson(path.join(projectDir, "config.json"), {});
  const providerFile = path.resolve(root, config.contextProviderFile || "../context-provider.js");
  if (!fs.existsSync(providerFile)) throw new Error(`context provider not found: ${providerFile}`);
  delete require.cache[require.resolve(providerFile)];
  const provider = require(providerFile);
  if (typeof provider.createApi !== "function") throw new Error("context provider must export createApi(projectRoot)");
  const api = provider.createApi(root) || {};
  if (typeof api.revision !== "string" || !api.revision) throw new Error("context provider API must include a non-empty revision");
  const sourceFile = path.join(projectDir, projectConfig.sourceFile || "translations.csv");
  return {
    root, projectDir, providerFile, provider, api, sourceFile,
    base: projectConfig.baseLanguage || config.baseCol || "en",
    contextColumn: config.contextColumn || "context",
    manualFile: path.resolve(root, config.contextsFile || path.join(projectDir, "contexts.json")),
    queueFile: path.resolve(root, config.glossaryQueueFile || path.join(projectDir, "terms_queue.json")),
    rejectedFile: path.resolve(root, config.glossaryRejectedFile || path.join(projectDir, "not_terms.json")),
    glossaryFile: path.resolve(root, config.glossaryFile || path.join(projectDir, "glossary.json")),
  };
}

const skeleton = (value) => String(value || "").replace(/\{[^}]*\}/g, "{}").trim();

function visit(value, callback, key = null, parent = null) {
  if (Array.isArray(value)) return value.forEach((item) => visit(item, callback, key, value));
  if (value && typeof value === "object") return Object.entries(value).forEach(([childKey, child]) => visit(child, callback, childKey, value));
  if (typeof value === "string") callback(value, key, parent);
}

function buildColumnIndex(state) {
  const exact = new Map(), bySkeleton = new Map();
  const put = (text, context) => {
    const value = String(text || "").trim(), note = String(context || "").trim();
    if (!value || !note) return;
    if (!exact.has(value)) exact.set(value, note);
    if (!bySkeleton.has(skeleton(value))) bySkeleton.set(skeleton(value), note);
  };
  for (const [file, columns] of Object.entries(state.provider.csv || {})) {
    if (typeof state.api.table !== "function") throw new Error(`provider API has no table() for ${file}`);
    for (const row of state.api.table(file) || []) for (const [column, make] of Object.entries(columns)) {
      if (typeof make === "function") put(row[column], make(row, state.api));
    }
  }
  for (const [file, keys] of Object.entries(state.provider.json || {})) {
    if (typeof state.api.json !== "function") throw new Error(`provider API has no json() for ${file}`);
    visit(state.api.json(file), (value, key, parent) => {
      const make = keys[key];
      if (typeof make === "function") put(value, make(value, parent || {}, state.api));
    });
  }
  return { exact, bySkeleton };
}

function resolver(state) {
  const manual = readJson(state.manualFile, {});
  const columns = buildColumnIndex(state);
  return (text) => {
    const candidates = [];
    if (manual[text]) candidates.push({ context: String(manual[text]), via: "manual" });
    const column = columns.exact.get(text) || columns.bySkeleton.get(skeleton(text));
    if (column) candidates.push({ context: column, via: "column" });
    const domain = typeof state.api.domainOf === "function" ? state.api.domainOf(text) : null;
    const make = domain && (state.provider.po || {})[domain];
    if (typeof make === "function") {
      const context = make(text, state.api, typeof state.api.hit === "function" ? state.api.hit(text) || {} : {});
      if (String(context || "").trim()) candidates.push({ context: String(context).trim(), via: `domain:${domain}` });
    }
    for (const rule of state.provider.rules || []) {
      if (!rule || !(rule.match instanceof RegExp) || typeof rule.context !== "function") continue;
      rule.match.lastIndex = 0;
      const matched = text.match(rule.match);
      if (!matched) continue;
      const context = rule.context(matched, state.api);
      if (String(context || "").trim()) candidates.push({ context: String(context).trim(), via: `rule:${rule.name || rule.match}` });
    }
    return candidates.length ? { ...candidates[0], alternatives: candidates.slice(1) } : null;
  };
}

function knownTerms(state) {
  const value = readJson(state.glossaryFile, {});
  return new Set([...Object.keys(value.terms || {}), ...Object.keys(value.manual || {})]);
}

function inspect(state) {
  const parsed = parseObjects(fs.readFileSync(state.sourceFile, "utf8"));
  if (!parsed.head.includes(state.base) || !parsed.head.includes(state.contextColumn)) throw new Error(`source CSV must contain ${state.base} and ${state.contextColumn}`);
  const resolve = resolver(state), known = knownTerms(state), deferred = readJson(state.queueFile, {});
  const counts = {}, classes = {}, forms = new Map(), changes = [], retained = [], pending = [], waiting = [], fixed = [], conflicts = [], masked = [];
  for (const row of parsed.rows) {
    const text = row[state.base];
    if (!text) continue;
    const result = resolve(text), had = String(row[state.contextColumn] || "").trim();
    if (result) counts[result.via] = (counts[result.via] || 0) + 1;
    if (result) {
      const key = `${result.via}\u0000${result.context}`;
      const form = forms.get(key) || { via: result.via, context: result.context, rows: 0, example: text };
      form.rows++;
      forms.set(key, form);
    }
    if (result && result.alternatives.length) {
      if (result.via === "manual") masked.push({ text, selected: result, alternatives: result.alternatives });
      if (new Set([result.context, ...result.alternatives.map((item) => item.context)]).size > 1) conflicts.push({ text, selected: result, alternatives: result.alternatives });
    }
    if (result && result.via === "manual" && result.context !== had) changes.push({ row, text, from: had, to: result.context, via: result.via });
    else if (result && !had) changes.push({ row, text, from: "", to: result.context, via: result.via });
    else if (had) retained.push({ text, current: had, proposed: result && result.context, via: result && result.via });
    if (!had && !result) {
      if (known.has(text)) fixed.push(text);
      else if (deferred[text]) waiting.push(text);
      else pending.push(text);
    }
    const className = typeof state.api.classOf === "function" ? String(state.api.classOf(text) || "(unclassified)") : "(all)";
    const outcome = result ? result.via : had ? "existing" : known.has(text) ? "glossary" : deferred[text] ? "waiting" : "pending";
    classes[className] ||= {};
    classes[className][outcome] = (classes[className][outcome] || 0) + 1;
  }
  return { ...parsed, counts, classes, formulations: [...forms.values()], changes, retained, pending, waiting, fixed, conflicts, masked };
}

function revision(state) {
  const hash = crypto.createHash("sha256");
  for (const file of [state.providerFile, state.sourceFile, state.manualFile]) {
    hash.update(fs.existsSync(file) ? fs.readFileSync(file) : "");
  }
  hash.update(state.api.revision);
  return hash.digest("hex").slice(0, 16);
}

function report(state) {
  const result = inspect(state);
  console.log(`rows: ${result.rows.length}; resolved: ${Object.values(result.counts).reduce((a, b) => a + b, 0)}; pending: ${result.pending.length}`);
  console.log(`resolution: ${Object.entries(result.counts).map(([key, value]) => `${key} ${value}`).join(", ") || "none"}`);
  console.log(`classes: ${Object.entries(result.classes).map(([name, values]) => `${name} ${Object.values(values).reduce((a, b) => a + b, 0)}`).join(", ")}`);
  console.log(`conflicts: ${result.conflicts.length}; masked by manual: ${result.masked.length}`);
  console.log(`kept existing: ${result.rows.length - result.changes.length - result.pending.length - result.waiting.length - result.fixed.length}; glossary: ${result.fixed.length}; waiting for glossary: ${result.waiting.length}`);
}

function apply(state, options) {
  const token = revision(state), result = inspect(state);
  if (!options.write) return console.log(JSON.stringify({ revision: token, changes: result.changes.length, by_source: result.counts, by_class: result.classes, formulations: result.formulations, retained: result.retained.slice(0, Number(options.examples || 8)), conflicts: result.conflicts, masked_by_manual: result.masked, examples: result.changes.slice(0, Number(options.examples || 8)).map(({ text, from, to, via }) => ({ text, from, to, via })) }, null, 2));
  if (!options.expect || options.expect !== token) throw new Error(`input revision changed; preview again (current ${token})`);
  for (const change of result.changes) change.row[state.contextColumn] = change.to;
  if (result.changes.length) writeCsv(state.sourceFile, result.head, result.rows);
  console.log(`written contexts: ${result.changes.length}`);
}

function refsOf(state, text) {
  const hit = typeof state.api.hit === "function" ? state.api.hit(text) || {} : {};
  return Array.isArray(hit.refs) ? hit.refs.map(String) : [];
}

function groupOf(state, text) {
  return (refsOf(state, text)[0] || "(ungrouped)").replace(/:\d+$/, "");
}

function next(state, options) {
  const result = inspect(state), groups = new Map();
  for (const text of result.pending) {
    const group = groupOf(state, text);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(text);
  }
  const ranked = [...groups].sort((a, b) => b[1].length - a[1].length);
  const picked = options.group ? ranked.find(([group]) => group === options.group) : ranked[0];
  if (!picked) return console.log(JSON.stringify({ done: true }, null, 2));
  console.log(JSON.stringify({ group: picked[0], strings: picked[1].map((text) => ({ text, refs: refsOf(state, text) })), groups_left: ranked.length, strings_left: result.pending.length, answer_shape: { group: picked[0], contexts: { "source text": "One English sentence." }, needs_glossary: {} } }, null, 2));
}

function submit(state, options) {
  if (!options.json) throw new Error("submit requires --json <answer-file>");
  const answer = readJson(path.resolve(options.json), null);
  if (!answer || !answer.group || (!answer.contexts && !answer.needs_glossary)) throw new Error("answer requires group and contexts and/or needs_glossary");
  const result = inspect(state), source = new Set(result.rows.map((row) => row[state.base]));
  const pending = new Set(result.pending);
  const manual = readJson(state.manualFile, {}), deferred = readJson(state.queueFile, {}), rejected = readJson(state.rejectedFile, {}), known = knownTerms(state);
  const problems = [];
  for (const [text, context] of Object.entries(answer.contexts || {})) {
    const value = String(context || "").trim();
    if (!source.has(text) || groupOf(state, text) !== answer.group) problems.push(`source is not in group: ${text}`);
    else if (known.has(text)) problems.push(`glossary term does not need row context: ${text}`);
    else if (!pending.has(text) && manual[text] !== value) problems.push(`source is not pending: ${text}`);
    else if (!value || value.length > 200) problems.push(`context must contain 1-200 characters: ${text}`);
    else manual[text] = value;
  }
  for (const [text, context] of Object.entries(answer.needs_glossary || {})) {
    const value = String(context || "").trim();
    if (!source.has(text) || groupOf(state, text) !== answer.group) problems.push(`source is not in group: ${text}`);
    else if (rejected[text]) problems.push(`already rejected as a term; provide context instead: ${text}`);
    else if (!pending.has(text) && deferred[text] !== value) problems.push(`source is not pending: ${text}`);
    else if (!value || value.length > 200) problems.push(`term evidence must contain 1-200 characters: ${text}`);
    else if (Object.hasOwn(answer.contexts || {}, text)) problems.push(`source is in contexts and needs_glossary: ${text}`);
    else deferred[text] = value;
  }
  if (problems.length) throw new Error(problems.join("\n"));
  writeJson(state.manualFile, Object.fromEntries(Object.entries(manual).sort(([a], [b]) => a.localeCompare(b))));
  writeJson(state.queueFile, Object.fromEntries(Object.entries(deferred).sort(([a], [b]) => a.localeCompare(b))));
  console.log(`recorded contexts: ${Object.keys(answer.contexts || {}).length}; sent to glossary: ${Object.keys(answer.needs_glossary || {}).length}`);
}

function main() {
  const options = args(process.argv), command = options._[0] || "report", state = load(options);
  if (command === "report") return report(state);
  if (command === "apply") return apply(state, options);
  if (command === "next") return next(state, options);
  if (command === "submit") return submit(state, options);
  throw new Error(`unknown command: ${command}`);
}

try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
