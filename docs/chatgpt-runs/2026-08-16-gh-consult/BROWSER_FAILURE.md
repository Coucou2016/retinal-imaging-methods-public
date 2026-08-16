# BROWSER_FAILURE — ChatGPT consult (2026-08-16-gh-consult)

**Date:** 2026-08-16  
**Goal:** New ChatGPT chat + web search ON; paste `TASK_BRIEF.md` (text only) + instruct reading https://github.com/Coucou2016/retinal-imaging-methods-public.

## Evidence

| Attempt | Result |
|---------|--------|
| `GetMcpTools` catalog | Only `cursor-app-control` ready. **No** `cursor-ide-browser` (no `browser_navigate` / tabs / lock). |
| `CallMcpTool` `open_resource` → `https://chatgpt.com/` | Failed: `Error: unknown agent: …` |
| `Start-Process https://chatgpt.com/` | Launched system default browser for the user (no agent control of that session). |
| Playwright Chromium headless → `https://chatgpt.com/` | Landed on Cloudflare interstitial: title **“Just a moment…”**, URL contains `__cf_chl_rt_tk=…`. No composer, no New chat, no web-search toggle. |
| Probe artifacts | `_probe_chatgpt.json`, `_probe_chatgpt.png`, `_probe_chatgpt.py` in this folder |

## What was **not** done

- No ChatGPT login / 2FA completed by the agent (blocked before login UI was usable).
- **No ChatGPT reply invented.**
- No ZIP / file upload to ChatGPT.

## User action (manual)

1. Open https://chatgpt.com/ → **New chat** → enable **web search** if available.  
2. Paste [`TASK_BRIEF.md`](TASK_BRIEF.md) (includes public GitHub URL; text only).  
3. Return chat URL + answers to Cursor for independent verify before further edits.
