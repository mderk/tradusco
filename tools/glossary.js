#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { parseObjects } = require("./lib/csv");

const MODES = new Set(["exact", "stem", "keep", "skip"]);
const TERM = /\b[A-Z][a-z]{2,}(?:\s+(?:of|the|and|to)\s+[A-Z]?[a-z]{2,}|\s+[A-Z][a-z]{2,})*/g;

function args(argv) {
  const out = { _: [] };
  for (let i = 2; i < argv.length; i++) {
    const value = argv[i];
    if (!value.startsWith("--")) out._.push(value);
    else {
      const key = value.slice(2);
      const next = argv[i + 1];
      const parsed = !next || next.startsWith("--") ? true : next;
      if (key === "translation") (out[key] ||= []).push(parsed);
      else out[key] = parsed;
      if (parsed !== true) i++;
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

function load(options) {
  const configFile = path.resolve(String(options.config || ".tradusco/config.json"));
  const root = path.dirname(configFile);
  const config = readJson(configFile, null);
  if (!config) throw new Error(`config not found: ${configFile}`);
  const projectDir = path.resolve(root, config.projectDir || "project");
  const projectConfig = readJson(path.join(projectDir, "config.json"), {});
  const glossaryFile = path.resolve(root, config.glossaryFile || path.join(projectDir, "glossary.json"));
  const sourceFile = path.join(projectDir, projectConfig.sourceFile || "translations.csv");
  const base = projectConfig.baseLanguage || config.baseCol || "en";
  const locales = config.locales || (projectConfig.languages || []).filter((lang) => lang !== base);
  const translate = config.translate || {};
  if (Object.hasOwn(translate, "regenerateLangs")) throw new Error("translate.regenerateLangs was replaced by translate.protectLangs; list protected locales instead");
  const protectedLocales = new Set((translate.protectLangs || []).map(String));
  return {
    config, root, projectDir, glossaryFile, sourceFile, base, locales,
    reviewed: locales.filter((lang) => protectedLocales.has(lang)),
    rejectedFile: path.resolve(root, config.glossaryRejectedFile || path.join(projectDir, "not_terms.json")),
    queueFile: path.resolve(root, config.glossaryQueueFile || path.join(projectDir, "terms_queue.json")),
    contextsFile: path.resolve(root, config.contextsFile || path.join(projectDir, "contexts.json")),
  };
}

function glossary(state) {
  const value = readJson(state.glossaryFile, { terms: {}, manual: {} });
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("glossary must be an object");
  value.terms ||= {};
  value.manual ||= {};
  return value;
}

function validateTerms(terms) {
  if (!terms || typeof terms !== "object" || Array.isArray(terms)) throw new Error("provider output must be an object keyed by term");
  for (const [term, entry] of Object.entries(terms)) {
    if (!term || !entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error(`invalid term entry: ${term}`);
    if (entry.mode && !MODES.has(entry.mode)) throw new Error(`invalid mode for ${term}: ${entry.mode}`);
    if (entry.t !== undefined && (!entry.t || typeof entry.t !== "object" || Array.isArray(entry.t))) throw new Error(`invalid translations for ${term}`);
  }
}

function run(command, cwd) {
  if (!Array.isArray(command) || !command.length || command.some((part) => typeof part !== "string")) throw new Error("glossarySourceCommand must be a non-empty argv array");
  const result = spawnSync(command[0], command.slice(1), { cwd, encoding: "utf8" });
  if (result.status !== 0) throw new Error((result.stderr || result.stdout || `provider exited ${result.status}`).trim());
}

function prepare(state, options) {
  const temporary = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "tradusco-glossary-")), "terms.json");
  try {
    run([...(state.config.glossarySourceCommand || []), "--output", temporary], state.root);
    const terms = readJson(temporary, null);
    validateTerms(terms);
    const current = glossary(state);
    const changed = new Set([...Object.keys(current.terms), ...Object.keys(terms)]);
    const count = [...changed].filter((term) => JSON.stringify(current.terms[term]) !== JSON.stringify(terms[term])).length;
    console.log(`generated terms: ${Object.keys(terms).length}; changed: ${count}`);
    if (options.write) {
      current.terms = Object.fromEntries(Object.entries(terms).sort(([a], [b]) => a.localeCompare(b)));
      writeJson(state.glossaryFile, current);
      console.log(`written: ${path.relative(state.root, state.glossaryFile)}`);
    } else console.log("preview only; add --write to apply");
  } finally {
    fs.rmSync(path.dirname(temporary), { recursive: true, force: true });
  }
}

function startsSentence(text, at) {
  const before = text.slice(0, at).replace(/["'“«_\s]+$/, "");
  return before === "" || /[.!?:;\n\r…—-]$/.test(before);
}

function candidates(state, minimum) {
  if (!fs.existsSync(state.sourceFile)) throw new Error(`source CSV not found: ${state.sourceFile}`);
  const rows = parseObjects(fs.readFileSync(state.sourceFile, "utf8")).rows;
  const known = new Set([...Object.keys(glossary(state).terms), ...Object.keys(glossary(state).manual)]);
  const rejected = readJson(state.rejectedFile, {});
  const deferred = readJson(state.queueFile, {});
  const stats = new Map();
  for (const row of rows) for (const match of String(row[state.base] || "").matchAll(TERM)) {
    const term = match[0].trim();
    const item = stats.get(term) || { total: 0, mid: 0, examples: [] };
    item.total++;
    if (!startsSentence(row[state.base], match.index)) item.mid++;
    if (item.examples.length < 6) item.examples.push(row[state.base]);
    stats.set(term, item);
  }
  const items = Object.entries(deferred)
    .filter(([term]) => !known.has(term) && !rejected[term])
    .map(([term, note]) => ({ term, source: "context", note }));
  for (const [term, item] of stats) {
    if (known.has(term) || rejected[term] || term in deferred) continue;
    if (item.total >= minimum && item.mid >= 3 && item.mid / item.total >= 0.2) items.push({ term, source: "frequency", ...item });
  }
  return items.sort((a, b) =>
    Number(b.source === "context") - Number(a.source === "context") || (b.mid || 0) - (a.mid || 0)
  );
}

function report(state, options) {
  const value = glossary(state);
  const items = candidates(state, Number(options.min || 6));
  const coverage = runPython(state, "--coverage");
  const gaps = Object.entries(value.manual).filter(([, entry]) =>
    state.reviewed.some((lang) => !(entry.t || {})[lang])
  );
  console.log(`glossary: terms ${Object.keys(value.terms).length}, manual ${Object.keys(value.manual).length}`);
  console.log(`coverage: unmatched ${coverage.filter((item) => !item.matched_rows).length}; whole-phrase only ${coverage.filter((item) => item.matched_rows && item.matched_rows === item.whole_phrase_rows).length}`);
  console.log(`candidate queue: ${items.length}; rejected: ${Object.keys(readJson(state.rejectedFile, {})).length}`);
  console.log(`manual entries missing reviewed-language values: ${gaps.length}`);
  if (options.check && (gaps.length || items.some((item) => item.source === "context"))) process.exitCode = 1;
}

function next(state, options) {
  const items = candidates(state, Number(options.min || 6));
  const item = options.term ? items.find(({ term }) => term === options.term) : items[0];
  if (!item) return console.log(JSON.stringify({ done: true }, null, 2));
  const rows = parseObjects(fs.readFileSync(state.sourceFile, "utf8")).rows;
  const samples = rows.filter((row) => new RegExp(`\\b${item.term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "i").test(row[state.base] || "")).slice(0, 8);
  const languages = Object.fromEntries(state.reviewed.map((lang) => [lang, samples.map((row) => ({ source: row[state.base], translation: row[lang] })).filter((sample) => sample.translation)]));
  console.log(JSON.stringify({ ...item, queue_left: items.length, examples: samples.map((row) => row[state.base]), languages, answer_shape: { term: item.term, entry: { mode: "stem", note: "One English sentence.", t: {} } } }, null, 2));
}

function runPython(state, ...extra) {
  const script = path.resolve(__dirname, "..", "glossary.py");
  const bundledPython = path.resolve(__dirname, "..", ".venv", "bin", "python");
  const python = process.env.PYTHON || (fs.existsSync(bundledPython) ? bundledPython : "python3");
  const result = spawnSync(python, [script, "--project-dir", state.projectDir, "--glossary", state.glossaryFile, "--json", ...extra], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (result.error) throw result.error;
  if (result.status !== 0 && result.status !== 1) throw new Error((result.stderr || `lint exited ${result.status}`).trim());
  if (!result.stdout.trim()) throw new Error((result.stderr || "lint returned no JSON").trim());
  return JSON.parse(result.stdout);
}

function lintFindings(state) {
  return runPython(state);
}

function lint(state, json = false) {
  const findings = lintFindings(state);
  if (json) console.log(JSON.stringify(findings, null, 2));
  else console.log(`glossary findings: ${findings.length}`);
  return findings.length ? 1 : 0;
}

function submit(state, options) {
  let answer;
  if (options.json) answer = readJson(path.resolve(options.json), null);
  else if (options.term && options.reject) answer = { term: options.term, not_a_term: options.reject };
  else if (options.term && options.mode && options.note && options.translation) {
    const translations = {};
    for (const item of options.translation) {
      const at = String(item).indexOf("=");
      if (at < 1 || at === String(item).length - 1) throw new Error("--translation must be LANG=VALUE");
      translations[String(item).slice(0, at)] = String(item).slice(at + 1);
    }
    answer = { term: options.term, entry: { mode: options.mode, note: options.note, t: translations } };
  } else throw new Error("submit requires --term with --reject, or --mode, --note and --translation LANG=VALUE; --json remains available for batch input");
  if (!answer || !answer.term || Boolean(answer.entry) === Boolean(answer.not_a_term)) throw new Error("answer requires term and exactly one of entry or not_a_term");
  const item = candidates(state, 1).find(({ term }) => term === answer.term);
  if (!item) {
    const recorded = answer.not_a_term
      ? readJson(state.rejectedFile, {})[answer.term] === String(answer.not_a_term)
      : JSON.stringify(glossary(state).manual[answer.term]) === JSON.stringify(answer.entry);
    if (!recorded) throw new Error(`term is not queued: ${answer.term}`);
    return finishDecision(state, answer);
  }
  if (answer.not_a_term) {
    const rejected = readJson(state.rejectedFile, {});
    rejected[answer.term] = String(answer.not_a_term);
    writeJson(state.rejectedFile, rejected);
  } else {
    validateTerms({ [answer.term]: answer.entry });
    if (!answer.entry.note || !answer.entry.t) throw new Error("entry requires note and t");
    const missing = state.reviewed.filter((lang) => !answer.entry.t[lang]);
    if (missing.length) throw new Error(`missing reviewed-language values: ${missing.join(", ")}`);
    const value = glossary(state);
    const backup = fs.existsSync(state.glossaryFile) ? fs.readFileSync(state.glossaryFile) : null;
    const before = lintFindings(state).length;
    value.manual[answer.term] = answer.entry;
    value.manual = Object.fromEntries(Object.entries(value.manual).sort(([a], [b]) => a.localeCompare(b)));
    writeJson(state.glossaryFile, value);
    try {
      const after = lintFindings(state).length;
      if (after <= before + 50) return finishDecision(state, answer);
      throw new Error(`glossary entry produced too many new findings: ${before} -> ${after}`);
    } catch (error) {
      if (backup) fs.writeFileSync(state.glossaryFile, backup);
      else fs.rmSync(state.glossaryFile, { force: true });
      throw error;
    }
  }
  finishDecision(state, answer);
}

function finishDecision(state, answer) {
  const deferred = readJson(state.queueFile, {});
  if (answer.not_a_term && deferred[answer.term]) {
    const contexts = readJson(state.contextsFile, {});
    contexts[answer.term] ||= deferred[answer.term];
    writeJson(state.contextsFile, Object.fromEntries(Object.entries(contexts).sort(([a], [b]) => a.localeCompare(b))));
  }
  delete deferred[answer.term];
  writeJson(state.queueFile, deferred);
  console.log(`recorded: ${answer.term}`);
}

function main() {
  const options = args(process.argv);
  const command = options._[0] || "report";
  const state = load(options);
  if (command === "prepare") return prepare(state, options);
  if (command === "report") return report(state, options);
  if (command === "next") return next(state, options);
  if (command === "submit") return submit(state, options);
  if (command === "lint") process.exitCode = lint(state, options.json);
  else throw new Error(`unknown command: ${command}`);
}

try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
