# adaptive_review_extractor_codegen.py
import os
import sys
import json
import time
import uuid
import shutil
import random
import logging
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
print(">>> SCRIPT STARTED <<<")

# =========================
# Config / Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
MAX_CODE_GENERATION_ATTEMPTS = 5
MAX_CONSECUTIVE_FAILURES = 3
BATCH_SIZE = 50
MAX_PAGES = 200  # safety cap

# =========================
# Bright Data config
# =========================
@dataclass
class BrightDataConfig:
    api_key: str
    zone: str = "web_unlocker1"     # your zone name
    country: str = "US"
    endpoint: str = "https://api.brightdata.com/request"
    render_js: bool = True
    timeout_s: int = 60
    retries: int = 3

def initialize_brightdata_proxy(cfg: BrightDataConfig):
    """
    Simple session-id pool for Bright Data. Web Unlocker handles IP/CAPTCHAs;
    session_id provides stickiness/rotation across requests.
    """
    class _Pool:
        def __init__(self):
            self.counter = 0
        def get_next(self) -> str:
            self.counter += 1
            return f"sess-{int(time.time())}-{self.counter}-{uuid.uuid4().hex[:6]}"
    return _Pool()

# =========================
# LLM plumbing (stubs you replace)
# =========================
def load_custom_prompt(tenant_id: str, target_url: str) -> str:
    # Customize per tenant/site as needed
    return """You are generating Python scraper code.
- Import `bd_fetch_html` from `bd_sdk` (we provide this module in the sandbox).
- Implement: run(url: str, page: int, limit: int, session_id: str) -> list[dict]
- Use bd_fetch_html(url, session_id=session_id) to fetch HTML (JS-rendered via Bright Data).
- Parse reviews using BeautifulSoup or lxml.
- Return list of dicts with keys: text, rating, date, author.
- Support pagination using `page` (e.g., ?page=2) when applicable.
- Respect `limit` to cap results in a single call.
- On errors or when nothing found, return [] (do not raise).
"""

def call_llm(prompt: str) -> str:
    # Demo stub returns an Amazon-style scraper.
    # In production, replace with a real LLM call.
    return '''\
import urllib.parse as _up
import re
from bs4 import BeautifulSoup
from bd_sdk import bd_fetch_html

def _txt(el): return el.get_text(" ", strip=True) if el else None
def _num(s):
    if not s: return None
    m = re.search(r"(\\d+(?:\\.\\d+)?)", s)
    try: return float(m.group(1)) if m else None
    except: return None

def _with_page_number(url: str, page: int) -> str:
    # Ensure we drive Amazon pagination via pageNumber=
    parsed = _up.urlparse(url)
    q = dict(_up.parse_qsl(parsed.query))
    q["pageNumber"] = str(page if page else 1)
    new_query = _up.urlencode(q)
    return _up.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

def run(url: str, page: int, limit: int, session_id: str):
    paged_url = _with_page_number(url, page or 1)
    html = bd_fetch_html(paged_url, session_id=session_id)
    soup = BeautifulSoup(html, "lxml")  # use lxml for robustness/speed

    out = []
    # Each review card:
    for box in soup.select("div[data-hook='review']"):
        text   = _txt(box.select_one("[data-hook='review-body']"))
        rating = _num(_txt(box.select_one("[data-hook='review-star-rating']")))
        date   = _txt(box.select_one("[data-hook='review-date']"))
        author = _txt(box.select_one("[data-hook='review-author']")) or _txt(box.select_one(".a-profile-name"))

        if text:
            out.append({"text": text, "rating": rating, "date": date, "author": author})
            if limit and len(out) >= limit:
                break

    return out
'''

def parse_code_from_response(code: str) -> str:
    # Hook to lint/scan the code if you want.
    return code

def create_fix_prompt(original_code: str, error: str, error_history: List[str], url: str) -> str:
    prev = ""
    if error_history:
        prev = "Previous errors:\\n" + "\\n".join(f"- {e}" for e in error_history[:-1])
    return f"""The following Python scraper failed.

URL: {url}

Original Code:
{original_code}

Error encountered:
{error}

{prev}

Please FIX the code to:
1) Keep using `from bd_sdk import bd_fetch_html`
2) Keep the function signature: run(url, page, limit, session_id) -> list[dict]
3) Maintain pagination (via 'page' param or by discovering 'next')
4) Be robust to missing DOM elements
Return ONLY the corrected Python code.
"""

# =========================
# Bright Data SDK injected into sandbox (requests-based; no curl)
# =========================
def _bd_sdk_source(cfg: BrightDataConfig) -> str:
    """
    Written into the sandbox so generated code can:
        from bd_sdk import bd_fetch_html
    This wraps Bright Data Web Unlocker with requests.post and returns HTML.
    """
    return f'''\
import os, json, time, random
import requests

BRIGHTDATA_API = "{cfg.endpoint}"
BRIGHTDATA_ZONE = "{cfg.zone}"
BRIGHTDATA_COUNTRY = "{cfg.country}"
BRIGHTDATA_RENDER = {str(cfg.render_js)}
BRIGHTDATA_TIMEOUT = {cfg.timeout_s}

API_KEY = os.getenv("BRIGHTDATA_API_KEY")
if not API_KEY:
    raise RuntimeError("BRIGHTDATA_API_KEY not set in sandbox")

def bd_fetch_html(url: str, session_id: str = None, retries: int = 3) -> str:
    payload = {{
        "zone": BRIGHTDATA_ZONE,
        "url": url,
        "format": "raw",
        "render": BRIGHTDATA_RENDER,
        "country": BRIGHTDATA_COUNTRY
    }}
    if session_id:
        # Some accounts expect "session_id" instead of "session".
        # If your account requires "session_id", swap the key below.
        payload["session"] = session_id
    headers = {{
        "Content-Type": "application/json",
        "Authorization": f"Bearer {{API_KEY}}"
    }}
    last_status = None
    for attempt in range(1, retries+1):
        r = requests.post(BRIGHTDATA_API, headers=headers, data=json.dumps(payload), timeout=BRIGHTDATA_TIMEOUT)
        if r.ok and r.text:
            return r.text
        last_status = r.status_code
        time.sleep(min(2 ** attempt, 8) + random.random())
    raise RuntimeError(f"Bright Data fetch failed after {{retries}} attempts (status={{last_status}}) for {{url}}")
'''

# =========================
# Sandbox execution utilities
# =========================
def create_sandbox_environment(code_src: str, bd_cfg: BrightDataConfig) -> str:
    """
    Creates a temp dir with:
      - scraper.py  (LLM code)
      - bd_sdk.py   (Bright Data wrapper)
      - runner.py   (entry to call scraper.run(...) and print JSON)
      - Installs deps into the temp folder so imports resolve.
    """
    workdir = tempfile.mkdtemp(prefix="scraper_sbx_")
    paths = {
        "scraper": os.path.join(workdir, "scraper.py"),
        "bd_sdk": os.path.join(workdir, "bd_sdk.py"),
        "runner": os.path.join(workdir, "runner.py")
    }
    with open(paths["scraper"], "w", encoding="utf-8") as f:
        f.write(code_src)
    with open(paths["bd_sdk"], "w", encoding="utf-8") as f:
        f.write(_bd_sdk_source(bd_cfg))

    runner_src = '''\
import os, sys, json, importlib.util

def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error":"missing args"}))
        sys.exit(0)
    url = sys.argv[1]
    page = int(sys.argv[2]) if sys.argv[2].isdigit() else 1
    limit = int(sys.argv[3]) if sys.argv[3].isdigit() else 50
    session_id = sys.argv[4]
    try:
        spec = importlib.util.spec_from_file_location("scraper", "scraper.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            print(json.dumps({"error":"no run() found"}))
            return
        out = mod.run(url, page, limit, session_id)
        if isinstance(out, list):
            print(json.dumps(out))
        elif out is None:
            pass
        else:
            print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
'''
    with open(paths["runner"], "w", encoding="utf-8") as f:
        f.write(runner_src)

    # --- Install deps into the sandbox folder so imports resolve there ---
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
         "--target", workdir, "requests", "beautifulsoup4", "lxml"],
        check=True
    )

    return workdir

def _read_json_from_stdout(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"error": f"Invalid JSON output: {stdout[:500]}"}

def execute_code_sandbox(code_src: str, url: str, session_id: str, bd_cfg: BrightDataConfig,
                         limit: int = 5, timeout_s: int = 30) -> Dict[str, Any]:
    """
    Compile a sandbox, run generated code once (page=1, small limit), validate shape.
    """
    workdir = create_sandbox_environment(code_src, bd_cfg)
    try:
        env = {
            "PYTHONPATH": workdir,
            "BRIGHTDATA_API_KEY": bd_cfg.api_key,  # only secret exposed to sandbox
        }
        cmd = ["python", os.path.join(workdir, "runner.py"), url, "1", str(limit), session_id]
        proc = subprocess.run(cmd, cwd=workdir, env={**os.environ, **env},
                              capture_output=True, text=True, timeout=timeout_s)
        stdout = proc.stdout.strip() or ""
        data = _read_json_from_stdout(stdout)

        if isinstance(data, dict) and "error" in data:
            return {"success": False, "error": data["error"]}

        if validate_review_structure(data):
            return {"success": True, "data": data}
        else:
            return {"success": False, "error": "Invalid output structure"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Sandbox timeout after {timeout_s}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def execute_code_page(code_src: str, url: str, page: int, session_id: str, bd_cfg: BrightDataConfig) -> Dict[str, Any]:
    """
    Execute generated code for a specific page, return {success, data?, error?}
    """
    workdir = create_sandbox_environment(code_src, bd_cfg)
    try:
        env = {"PYTHONPATH": workdir, "BRIGHTDATA_API_KEY": bd_cfg.api_key}
        cmd = ["python", os.path.join(workdir, "runner.py"), url, str(page), "50", session_id]
        proc = subprocess.run(cmd, cwd=workdir, env={**os.environ, **env},
                              capture_output=True, text=True, timeout=bd_cfg.timeout_s)
        stdout = proc.stdout.strip() or ""
        data = _read_json_from_stdout(stdout)

        if isinstance(data, dict) and "error" in data:
            return {"success": False, "error": data["error"]}

        ok = validate_review_structure(data)
        return {"success": ok, "data": data if ok else None, "error": None if ok else "Invalid output structure"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def validate_review_structure(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    # minimally require non-empty "text"; other fields may be None on some sites
    for r in data:
        if not isinstance(r, dict): return False
        if not r.get("text"): return False
    return True

# =========================
# Retry helper
# =========================
def execute_with_retry(code_src: str, url: str, page: int, session_id: str,
                       bd_cfg: BrightDataConfig, max_retries: int = 3) -> List[Dict]:
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        res = execute_code_page(code_src, url, page, session_id, bd_cfg)
        if res["success"]:
            return res["data"]
        time.sleep(delay)
        delay = min(delay * 2, 8)
    raise RuntimeError(f"execute_with_retry failed after {max_retries} attempts")

# =========================
# Core algorithm (your pseudocode)
# =========================
def generate_scraper_code(prompt_template: str, url: str) -> str:
    prompt = f"""{prompt_template}

Target URL: {url}

Requirements:
- Handle pagination automatically where applicable
- Extract fields: text, rating, date, author
- Use CSS selectors or XPath
- Include error handling
- Support 'page' param if present
- Detect end (empty result when done)
"""
    code = call_llm(prompt)
    return parse_code_from_response(code)

def attempt_recovery(current_code: str, url: str, page_number: int, error: str,
                     error_history: List[str], bd_cfg: BrightDataConfig) -> Dict[str, Any]:
    fix_prompt = create_fix_prompt(current_code, error, error_history, url)
    new_code = call_llm(fix_prompt)
    test_result = execute_code_sandbox(new_code, url, session_id=f"sess-recover-{uuid.uuid4().hex[:6]}",
                                       bd_cfg=bd_cfg, limit=5)
    return {"success": test_result["success"], "new_code": new_code if test_result["success"] else None}

def extract_all_reviews(working_code: str, url: str, proxy_pool, bd_cfg: BrightDataConfig,
                        error_history: List[str]) -> Tuple[List[Dict], str]:
    all_reviews: List[Dict] = []
    page_number = 1
    consecutive_failures = 0
    consecutive_empty_pages = 0
    has_more_pages = True

    while has_more_pages:
        if page_number > MAX_PAGES:
            logging.info(f"Reached MAX_PAGES={MAX_PAGES}. Stopping.")
            break
        try:
            current_proxy_session = proxy_pool.get_next()
            page_reviews = execute_with_retry(working_code, url, page_number, current_proxy_session, bd_cfg, max_retries=3)

            if not page_reviews:
                if consecutive_empty_pages > 2:
                    has_more_pages = False
                    break
                consecutive_empty_pages += 1
            else:
                consecutive_empty_pages = 0
                all_reviews.extend(page_reviews)
                logging.info(f"Extracted {len(page_reviews)} reviews from page {page_number} (total={len(all_reviews)})")

            page_number += 1
            consecutive_failures = 0
            time.sleep(random.uniform(1.0, 3.0))  # polite pacing

        except Exception as e:
            consecutive_failures += 1
            logging.error(f"Failed on page {page_number}: {e}")
            error_history.append(str(e))

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.error("Max consecutive failures reached. Attempting recovery...")
                rec = attempt_recovery(working_code, url, page_number, str(e), error_history, bd_cfg)
                if rec["success"] and rec["new_code"]:
                    working_code = rec["new_code"]
                    consecutive_failures = 0
                    logging.info("Recovery succeeded. Retrying page with repaired code.")
                    continue  # retry same page with fixed code
                else:
                    logging.error("Recovery failed. Stopping extraction.")
                    break

    return all_reviews, working_code

# =========================
# Persistence & Logging
# =========================
def save_reviews(reviews: List[Dict], tenant_id: str):
    os.makedirs("out", exist_ok=True)
    path = os.path.join("out", f"{tenant_id}_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(reviews)} reviews -> {path}")

def log(message: str): logging.info(message)
def log_error(message: str): logging.error(message)

# =========================
# Orchestration (main)
# =========================
def main(tenant_id: str, target_url: str, bd_cfg: BrightDataConfig) -> List[Dict]:
    # Initialize
    prompt_template = load_custom_prompt(tenant_id, target_url)
    proxy_pool = initialize_brightdata_proxy(bd_cfg)
    error_history: List[str] = []

    # Generate initial code
    scraper_code = generate_scraper_code(prompt_template, target_url)

    # Self-healing loop (test → regenerate up to N times)
    working_code: Optional[str] = None
    attempts = 0
    while working_code is None and attempts < MAX_CODE_GENERATION_ATTEMPTS:
        attempts += 1
        log(f"[codegen] test attempt {attempts}")
        test_result = execute_code_sandbox(scraper_code, target_url, session_id=proxy_pool.get_next(),
                                           bd_cfg=bd_cfg, limit=5)

        if test_result["success"]:
            working_code = scraper_code
            break
        else:
            err = test_result.get("error", "Unknown error")
            error_history.append(err)
            fix_prompt = create_fix_prompt(scraper_code, err, error_history, target_url)
            scraper_code = call_llm(fix_prompt)
            log_error(f"[codegen] attempt {attempts} failed: {err} -> regenerating")

    if working_code is None:
        raise RuntimeError(f"Failed to generate working scraper after {MAX_CODE_GENERATION_ATTEMPTS} attempts")

    # Extract all pages (with in-run recovery)
    all_reviews, working_code = extract_all_reviews(working_code, target_url, proxy_pool, bd_cfg, error_history)

    # Persist in batches
    if all_reviews:
        for i in range(0, len(all_reviews), BATCH_SIZE):
            save_reviews(all_reviews[i:i+BATCH_SIZE], tenant_id)

    return all_reviews

# =========================
# CLI entrypoint
# =========================
if __name__ == "__main__":
    TENANT_ID = os.getenv("TENANT_ID", "demo-tenant")
    TARGET_URL = os.getenv("TARGET_URL", "https://www.example.com/product/reviews")
    BD_KEY = os.getenv("BRIGHTDATA_API_KEY")
    if not BD_KEY:
        raise SystemExit("Set BRIGHTDATA_API_KEY in your environment.")

    bd_cfg = BrightDataConfig(
        api_key=BD_KEY,
        zone=os.getenv("BRIGHTDATA_ZONE", "web_unlocker1"),
        country=os.getenv("BRIGHTDATA_COUNTRY", "US"),
        endpoint=os.getenv("BRIGHTDATA_ENDPOINT", "https://api.brightdata.com/request"),
        render_js=True,
        timeout_s=60,
        retries=3
    )

    out = main(TENANT_ID, TARGET_URL, bd_cfg)
    logging.info(f"[DONE] total reviews gathered: {len(out)}")
