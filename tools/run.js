#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { parseObjects } = require("./lib/csv");

function args(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const value = argv[i];
    if (!value.startsWith("--")) continue;
    const key = value.slice(2), next = argv[i + 1];
    out[key] = !next || next.startsWith("--") ? true : next;
    if (out[key] !== true) i++;
  }
  return out;
}

function readJson(file, fallback = null) {
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : fallback;
}

function environment(file) {
  const env = { ...process.env };
  if (!file || !fs.existsSync(file)) return env;
  for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    const match = line.match(/^\s*([^#=\s]+)\s*=\s*(.*)\s*$/);
    if (!match) continue;
    let value = match[2];
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1);
    if (!(match[1] in env)) env[match[1]] = value;
  }
  return env;
}

function load(options) {
  const configFile = path.resolve(String(options.config || ".tradusco/config.json"));
  const root = path.dirname(configFile), config = readJson(configFile);
  if (!config) throw new Error(`config not found: ${configFile}`);
  const traduscoRoot = path.resolve(root, config.traduscoRoot || path.join(__dirname, ".."));
  const bundledPython = path.join(traduscoRoot, ".venv", "bin", "python");
  return {
    options, configFile, root, config, traduscoRoot,
    projectDir: path.resolve(root, config.projectDir || "project"),
    sourceCsv: path.resolve(root, config.sourceCsv || "translations.csv"),
    base: config.baseCol || "en",
    python: config.pythonCommand || (fs.existsSync(bundledPython) ? bundledPython : "python3"),
    env: environment(path.resolve(root, config.envFile || ".env.tradusco")),
  };
}

function commandLine(command) {
  return command.map((part) => (/\s/.test(part) ? JSON.stringify(part) : part)).join(" ");
}

function run(state, command, { capture = false } = {}) {
  console.log(`$ ${commandLine(command)}`);
  if (state.options["dry-run"]) return "";
  const result = spawnSync(command[0], command.slice(1), {
    cwd: state.root,
    env: state.env,
    encoding: "utf8",
    stdio: capture ? "pipe" : "inherit",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error((result.stderr || result.stdout || `command exited ${result.status}`).trim());
  if (capture && result.stdout) process.stdout.write(result.stdout);
  return result.stdout || "";
}

function stage(state, name, skipped, action) {
  if (skipped) return console.log(`${name}: skipped`);
  console.log(`${name}: start`);
  action();
  console.log(`${name}: done`);
}

function syncGlossary(state) {
  if (!state.config.glossaryFile || state.options["dry-run"]) return;
  const source = path.resolve(state.root, state.config.glossaryFile);
  if (!fs.existsSync(source)) throw new Error(`glossary not found: ${source}`);
  fs.mkdirSync(state.projectDir, { recursive: true });
  const target = path.join(state.projectDir, "glossary.json"), temporary = `${target}.tmp`;
  fs.copyFileSync(source, temporary);
  fs.renameSync(temporary, target);
}

function status(state) {
  const config = readJson(path.join(state.projectDir, "config.json"), {});
  const file = path.join(state.projectDir, config.sourceFile || "translations.csv");
  if (!fs.existsSync(file)) return console.log("status: project source not synced");
  const rows = parseObjects(fs.readFileSync(file, "utf8")).rows.filter((row) => row[config.baseLanguage || state.base]);
  const languages = (config.languages || state.config.locales || []).filter((lang) => lang !== (config.baseLanguage || state.base));
  const counts = Object.fromEntries(languages.map((lang) => [lang, rows.filter((row) => String(row[lang] || "").trim()).length]));
  const onlyFile = state.options["only-keys-file"] && path.resolve(state.root, String(state.options["only-keys-file"]));
  const onlyValue = onlyFile ? readJson(onlyFile, null) : null;
  if (onlyFile && (!Array.isArray(onlyValue) || onlyValue.some((key) => typeof key !== "string" || !key))) throw new Error("--only-keys-file must contain a JSON array of non-empty strings");
  const only = onlyValue && new Set(onlyValue);
  const selected = rows.filter((row) => only ? only.has(row[config.baseLanguage || state.base]) : languages.some((lang) => !String(row[lang] || "").trim())).map((row) => row[config.baseLanguage || state.base]);
  console.log(`status: rows ${rows.length}; ${Object.entries(counts).map(([lang, count]) => `${lang} ${count}/${rows.length}`).join(", ")}`);
  console.log(`selected: ${selected.length}${selected.length ? `; ${selected.slice(0, 20).map(JSON.stringify).join(", ")}` : ""}`);
}

function verifyArtifacts(state) {
  const command = state.config.artifactKeysCommand;
  if (!command) return;
  const output = run(state, command, { capture: true });
  if (state.options["dry-run"]) return;
  const keys = JSON.parse(output), expected = parseObjects(fs.readFileSync(state.sourceCsv, "utf8")).rows.map((row) => row[state.base]).filter(Boolean);
  if (!Array.isArray(keys) || keys.some((key) => typeof key !== "string")) throw new Error("artifactKeysCommand must output a JSON array of source keys");
  const present = new Set(keys), missing = expected.filter((key) => !present.has(key));
  if (missing.length) throw new Error(`delivery artifact missing keys: ${missing.slice(0, 20).map(JSON.stringify).join(", ")}`);
}

function main() {
  const state = load(args(process.argv)), options = state.options, config = state.config;
  const lock = path.join(state.projectDir, ".run.lock");
  if (!options["dry-run"]) {
    fs.mkdirSync(state.projectDir, { recursive: true });
    try { fs.writeFileSync(lock, `${process.pid}\n`, { flag: "wx" }); }
    catch { throw new Error(`project is locked: ${lock}`); }
    state.env.TRADUSCO_LOCK_PID = String(process.pid);
  }
  try {
    status(state);
    stage(state, "extract", options["skip-extract"], () => {
      for (const command of config.extractCommands || []) run(state, command);
    });
    stage(state, "sync", options["skip-sync"], () => run(state, [state.python, path.join(state.traduscoRoot, "sync_project_from_csv.py"), "--project-dir", state.projectDir, "--source-csv", state.sourceCsv, "--base-col", state.base]));
    stage(state, "glossary", options["skip-glossary"] || !config.glossarySourceCommand, () => {
      run(state, [process.execPath, path.join(state.traduscoRoot, "tools", "glossary.js"), "prepare", "--write", "--config", state.configFile]);
      syncGlossary(state);
    });
    if (!options["skip-glossary"] && !config.glossarySourceCommand) syncGlossary(state);
    stage(state, "context", options["skip-context"] || !config.contextProviderFile, () => {
      const preview = run(state, [process.execPath, path.join(state.traduscoRoot, "tools", "context.js"), "apply", "--config", state.configFile], { capture: true });
      if (!options["dry-run"]) {
        const revision = JSON.parse(preview).revision;
        run(state, [process.execPath, path.join(state.traduscoRoot, "tools", "context.js"), "apply", "--write", "--expect", revision, "--config", state.configFile]);
      }
    });
    stage(state, "translate", options["skip-translate"], () => {
      const translate = config.translate || {};
      const locales = String(options.langs || options.lang || (config.locales || []).join(","));
      if (!locales) throw new Error("no translation locales selected");
      if (options.regenerate) {
        const allowed = new Set((translate.regenerateLangs || []).map(String));
        const forbidden = locales.split(",").filter((lang) => !allowed.has(lang));
        if (forbidden.length) throw new Error(`regeneration is not allowed for: ${forbidden.join(", ")}`);
      }
      run(state, [state.python, "-u", path.join(state.traduscoRoot, "translate.py"), "-p", state.projectDir, "-l", locales, "-m", String(options.model || translate.model || "gemini"), "--method", String(translate.method || "auto"), "-b", String(translate.batchSize || 50), "--batch-max-input-tokens", String(translate.batchMaxInputTokens || 65536), "--request-timeout", String(translate.requestTimeout || 120), "-r", String(translate.retries ?? 3), "-d", String(translate.delaySeconds ?? 1), ...(translate.referenceLangs && translate.referenceLangs.length ? ["--reference-langs", translate.referenceLangs.join(",")] : []), ...(options.regenerate ? ["--regenerate"] : []), ...(options["only-keys-file"] ? ["--only-keys-file", path.resolve(state.root, String(options["only-keys-file"]))] : [])]);
    });
    stage(state, "audit", options["skip-audit"], () => run(state, [state.python, path.join(state.traduscoRoot, "audit_translations.py"), "--project-dir", state.projectDir]));
    stage(state, "delivery", options["skip-delivery"], () => {
      const review = path.join(state.traduscoRoot, "review_translations.py");
      const preview = run(state, [state.python, review, "export", "--config", state.configFile], { capture: true });
      if (!options["dry-run"]) run(state, [state.python, review, "export", "--write", "--expect", JSON.parse(preview).revision, "--config", state.configFile]);
      for (const command of config.deliveryCommands || []) run(state, command);
      verifyArtifacts(state);
    });
    status(state);
  } finally {
    if (!options["dry-run"]) fs.rmSync(lock, { force: true });
  }
}

try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
