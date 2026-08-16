# ChatGPT browser handoff failure — 2026-08-15 follow-up

## Attempts (stopped after few; continued with local engineering)

| # | Action | Result |
|---|--------|--------|
| 1 | `browser_tabs` list | Open tabs: (empty) |
| 2 | `browser_navigate` → https://chatgpt.com | `No browser tab available. Please navigate to a page first.` |
| 3 | `browser_navigate` with `position: 0` | Same: no browser tab available |
| 4 | `browser_tabs` action=`new` | Created `viewId=ee3545` (`about:blank`) |
| 5 | `browser_navigate` with that `viewId` | `Browser view not found: ee3545` |
| 6 | `browser_navigate` without viewId | No browser tab available |
| 7 | `browser_tabs` action=`new` again | Created `viewId=f5bafb` then vanished |
| 8 | `browser_tabs` select index 0 | `Tab 0 not found` |
| 9 | `browser_tabs` new `position=active` | Created `viewId=92d35c` |
| 10 | Immediate navigate + list | Navigate failed; **list empty** — tab vanished |

## Diagnosis

Same failure mode as audit run e3f38146: MCP can create a transient tab metadata object, but the Browser view disappears before navigation/lock/snapshot. Not a ChatGPT login/2FA blocker — the tab never stays alive long enough to load chatgpt.com.

## ChatGPT URL

**None** — handoff not started. Manual paste brief: update `../2026-08-15-audit/TASK_BRIEF.md` for remaining gaps if browser is fixed later.
