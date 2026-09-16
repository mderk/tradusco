"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

exports.createApi = (projectRoot) => {
  const file = path.resolve(projectRoot, "../source/messages.json");
  const source = fs.readFileSync(file, "utf8");
  const messages = JSON.parse(source);
  return {
    revision: crypto.createHash("sha256").update(source).digest("hex"),
    json: () => messages,
    hit: (text) => {
      const row = messages.find((item) => item.text === text);
      return row ? { refs: [`source/messages.json:${row.id}`] } : {};
    },
    classOf: () => "product-message",
  };
};

exports.json = {
  messages: {
    text: (_text, message) => message.context,
  },
};
