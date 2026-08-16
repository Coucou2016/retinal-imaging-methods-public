from playwright.sync_api import sync_playwright
import json, time, os

out_dir = r"E:\Projects\20260522-retinal-imaging\docs\chatgpt-runs\2026-08-16-gh-consult"
os.makedirs(out_dir, exist_ok=True)
out = {
  "url": None,
  "title": None,
  "login_signals": [],
  "has_composer": False,
  "has_new_chat": False,
  "has_search_toggle": False,
  "snippet": "",
  "error": None,
}

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.goto("https://chatgpt.com/", wait_until="domcontentloaded", timeout=60000)
        time.sleep(5)
        out["url"] = page.url
        out["title"] = page.title()
        html = page.content()
        text = page.inner_text("body")[:4000]
        out["snippet"] = text[:1500]
        lower = (html + "\n" + text).lower()
        for sig in [
            "log in", "sign up", "sign in", "create an account",
            "verify", "two-factor", "2fa", "authentication app",
            "enter code", "passkey", "continue with google",
            "welcome back", "email address",
        ]:
            if sig in lower:
                out["login_signals"].append(sig)
        out["has_composer"] = bool(page.query_selector('textarea, div[contenteditable="true"]'))
        out["has_new_chat"] = ("new chat" in lower)
        out["has_search_toggle"] = ("web search" in lower)
        page.screenshot(path=os.path.join(out_dir, "_probe_chatgpt.png"), full_page=False)
        with open(os.path.join(out_dir, "_probe_chatgpt.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        browser.close()
except Exception as e:
    out["error"] = repr(e)
    with open(os.path.join(out_dir, "_probe_chatgpt.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

print(json.dumps(out, ensure_ascii=False, indent=2))
