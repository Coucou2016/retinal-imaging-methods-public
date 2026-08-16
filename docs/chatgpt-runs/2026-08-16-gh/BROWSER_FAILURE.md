# BROWSER_FAILURE — ChatGPT automation

**Date:** 2026-08-16  
**Goal:** Open ChatGPT, enable web search if UI allows, paste TASK_BRIEF + public GitHub URL.

## What failed

1. MCP server `cursor-ide-browser` was **not available** in this agent session (`GetMcpTools` returned only `cursor-app-control`).
2. Therefore no `browser_navigate` / `browser_tabs` / lock workflow could run.
3. Prior paper-run notes (2026-08-16-paper) already documented unstable “No browser tab available” failures when the browser MCP was present.

## Fallback executed

- `Start-Process https://chatgpt.com/` (system default browser for user)
- `Start-Process https://github.com/Coucou2016/retinal-imaging-methods-public`
- Persisted paste-ready [`TASK_BRIEF.md`](TASK_BRIEF.md) with GitHub URL (text only; no ZIP upload)

## User action

1. In ChatGPT, start a new chat; turn **web search ON** if available.
2. Paste `TASK_BRIEF.md` contents.
3. Ask ChatGPT to read the public repo URL (do not upload private data).
4. Return answers to Cursor for independent verify before applying edits.
