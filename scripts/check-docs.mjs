#!/usr/bin/env node
/**
 * Documentation gate.
 *
 * Two checks that break quietly and only get noticed by a reader:
 *
 *   1. Every ```mermaid block must parse. Diagram syntax is not covered by any
 *      compiler, so a typo ships as a broken image on GitHub.
 *   2. Every relative link and image must resolve on disk.
 *
 * Parsing uses mermaid's own grammar under jsdom, so no browser is required.
 * Exit code 1 on any failure, with file:line for each.
 */
import { readFile, readdir } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// mermaid touches `document`/`window` during parsing; jsdom provides them.
const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", {
  value: dom.window.navigator,
  configurable: true,
});

const mermaid = (await import("mermaid")).default;
mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });

const MARKDOWN_ROOTS = ["README.md", "docs"];

async function collectMarkdown(target) {
  const abs = join(ROOT, target);
  const entries = await readdir(abs, { withFileTypes: true }).catch(() => null);
  if (!entries) return [abs];
  const nested = await Promise.all(
    entries.map((entry) => {
      const child = join(target, entry.name);
      if (entry.isDirectory()) return collectMarkdown(child);
      return entry.name.endsWith(".md") ? [join(ROOT, child)] : [];
    }),
  );
  return nested.flat();
}

/** Fenced code blocks with their opening-fence line number. */
function fencedBlocks(text, lang) {
  const lines = text.split("\n");
  const blocks = [];
  let open = null;
  for (let i = 0; i < lines.length; i += 1) {
    const fence = lines[i].match(/^\s*(`{3,}|~{3,})\s*([\w-]*)\s*$/);
    if (!open && fence) {
      open = { marker: fence[1][0], length: fence[1].length, lang: fence[2], line: i + 1 };
      continue;
    }
    if (open && fence && fence[1][0] === open.marker && fence[1].length >= open.length) {
      if (open.lang === lang) {
        blocks.push({ line: open.line, body: lines.slice(open.index ?? open.line, i).join("\n") });
      }
      open = null;
      continue;
    }
    if (open && open.index === undefined) open.index = i;
  }
  return blocks;
}

/** [text](target) and ![alt](target), ignoring inline code spans. */
function links(text) {
  const found = [];
  const lines = text.split("\n");
  let inFence = null;
  lines.forEach((line, i) => {
    const fence = line.match(/^\s*(`{3,}|~{3,})/);
    if (fence) {
      inFence = inFence ? null : fence[1][0];
      return;
    }
    if (inFence) return;
    const stripped = line.replace(/`[^`]*`/g, "");
    const re = /!?\[[^\]]*\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)/g;
    let m;
    while ((m = re.exec(stripped)) !== null) found.push({ line: i + 1, target: m[1] });
  });
  return found;
}

const problems = [];
const files = (await Promise.all(MARKDOWN_ROOTS.map(collectMarkdown))).flat().sort();
let diagramCount = 0;
let linkCount = 0;

for (const file of files) {
  const rel = relative(ROOT, file);
  const text = await readFile(file, "utf8");

  for (const block of fencedBlocks(text, "mermaid")) {
    diagramCount += 1;
    try {
      await mermaid.parse(block.body);
    } catch (err) {
      const detail = String(err?.message ?? err)
        .split("\n")
        .slice(0, 4)
        .join(" | ");
      problems.push(`${rel}:${block.line} mermaid parse error: ${detail}`);
    }
  }

  for (const { line, target } of links(text)) {
    const bare = target.split("#")[0];
    if (!bare || /^(https?:|mailto:|tel:|data:)/.test(bare)) continue;
    linkCount += 1;
    try {
      await readFile(resolve(dirname(file), bare));
    } catch {
      problems.push(`${rel}:${line} broken link: ${target}`);
    }
  }
}

if (problems.length) {
  console.error(`docs check failed (${problems.length} problem${problems.length === 1 ? "" : "s"}):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}

console.log(
  `docs check passed: ${diagramCount} mermaid diagrams parsed, ${linkCount} relative links resolved (${files.length} files)`,
);
