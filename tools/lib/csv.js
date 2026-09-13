"use strict";

function parse(text) {
  const table = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(field); field = ""; }
    else if (char === "\n") { row.push(field); table.push(row); row = []; field = ""; }
    else if (char !== "\r") field += char;
  }
  if (field || row.length) { row.push(field); table.push(row); }
  return table;
}

function parseObjects(text) {
  const table = parse(text);
  const head = table.shift() || [];
  return {
    head,
    rows: table.filter((row) => row.length > 1).map((row) =>
      Object.fromEntries(head.map((name, index) => [name, row[index] || ""]))
    ),
  };
}

function stringify(rows) {
  const cell = (value) => `"${String(value == null ? "" : value).replace(/"/g, '""')}"`;
  return rows.map((row) => row.map(cell).join(",")).join("\n");
}

module.exports = { parse, parseObjects, stringify };
