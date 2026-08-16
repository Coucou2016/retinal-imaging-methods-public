# BROWSER_FAILURE — ChatGPT iter-5 (2026-08-16)

**Goal:** ≥5 Cursor↔ChatGPT rounds with web search; paste GitHub URL each round.

## Evidence

| Attempt | Result |
|---------|--------|
| MCP `cursor-ide-browser` | **Absent** (only `cursor-app-control` ready) |
| `open_resource` → chatgpt.com | Failed: unknown agent |
| `Start-Process https://chatgpt.com/` | Opened system browser twice (start + mid) — no agent control |
| Playwright probe 1 (download shell) | Interrupted / hung with HF lock; see prior runs |
| Playwright probe 2 | URL contains `__cf_chl_rt_tk`; title **Just a moment...**; screenshot `_probe2.png`; JSON `_probe2.json` |

## What was done instead

- Five **local structured review rounds** with WebSearch literature verification: `round-1.md` … `round-5.md`
- Paste briefs for manual ChatGPT: `PASTE_BRIEF_R1.md` … `PASTE_BRIEF_R5.md`
- **No ChatGPT reply invented**

## User action

Paste each `PASTE_BRIEF_R*.md` into a New Chat (web search ON), return chat URLs into the matching `round-N.md` header, then ask Cursor to re-verify before further edits.
