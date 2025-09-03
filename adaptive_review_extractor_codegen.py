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


# Config / Logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
MAX_CODE_GENERATION_ATTEMPTS = 5
MAX_CONSECUTIVE_FAILURES = 3
BATCH_SIZE = 50
MAX_PAGES = 200
SAMPLE_HTML_MAX_CHARS = 150_000
SAMPLE_HTML_MIN_LEN = 512


# Bright Data config (HTTP API for Web Unlocker)

@dataclass
class BrightDataConfig:
    api_key: str
    zone: str = "web_unlocker1"
    country: str = "US"
    endpoint: str = "https://api.brightdata.com/request"
    render_js_hint: bool = True   # hint for server-side render param
    timeout_s: int = 40
    retries: int = 3

def initialize_brightdata_proxy(cfg: BrightDataConfig):
    class _Pool:
        def __init__(self):
            self.i = 0
            self.sessions = [f"sess-{int(time.time())}-{k}-{uuid.uuid4().hex[:6]}" for k in range(10)]
        def get_next(self) -> str:
            self.i += 1
            return self.sessions[self.i % len(self.sessions)]
    return _Pool()


# Support module sources we drop into sandbox:
#  - bd_sdk.py       (HTTP API fetcher)
#  - renderers.py    (Selenium headless renderer with proxy)
#  - shields.py      (robot/CAPTCHA detection)
#  - runner.py       (exec harness)

def _bd_sdk_source(cfg: BrightDataConfig) -> str:
    return f'''\
import requests, time, random

API_ENDPOINT = "{cfg.endpoint}"
API_KEY = "{cfg.api_key}"
ZONE = "{cfg.zone}"
COUNTRY = "{cfg.country}"
TIMEOUT = {cfg.timeout_s}
RENDER_HINT = {str(cfg.render_js_hint)}

def _ua():
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]
    return random.choice(uas)

def bd_fetch_html(url: str, session_id: str = None, render: bool = None, retries: int = 3) -> str:
    payload = {{
        "zone": ZONE,
        "url": url,
        "format": "raw",
        "render": (RENDER_HINT if render is None else render),
        "country": COUNTRY,
        "headers": {{
            "User-Agent": _ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }}
    }}
    if session_id:
        payload["session"] = session_id

    headers = {{
        "Content-Type": "application/json",
        "Authorization": f"Bearer {{API_KEY}}"
    }}

    last = None
    for attempt in range(1, (retries or 1) + 1):
        try:
            r = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT)
            last = r.text
            if r.status_code == 200 and last and len(last) > 100:
                low = last.lower()
                # quick shield sniff (coarse)
                if any(k in low for k in ["captcha", "robot check", "are you a robot", "verify you are human", "enable cookies"]):
                    time.sleep(2 ** attempt)
                    continue
                return last
            if r.status_code in (401,):
                raise RuntimeError("Bright Data auth failed (401)")
            if r.status_code in (429, 403, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            time.sleep(1)
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException:
            time.sleep(1.5)
    raise RuntimeError(f"Web Unlocker failed for {{url}} after {{retries}} attempts; last_len={{len(last) if last else 0}}")
'''

def _renderers_source() -> str:
    return r'''\
import os, time

def _get_proxy_caps():
    """
    Build Selenium proxy from env:
    PROXY_HOST, PROXY_PORT, PROXY_USERNAME, PROXY_PASSWORD
    Returns (seleniumwire_options, http_proxy_url or None)
    """
    host = os.getenv("PROXY_HOST")
    port = os.getenv("PROXY_PORT")
    user = os.getenv("PROXY_USERNAME")
    pwd  = os.getenv("PROXY_PASSWORD")
    if not host or not port:
        return None, None
    auth = f"{user}:{pwd}@" if (user and pwd) else ""
    proxy_url = f"http://{auth}{host}:{port}"
    # seleniumwire
    sw_opts = {"proxy": {"http": proxy_url, "https": proxy_url, "no_proxy": "localhost,127.0.0.1"}}
    return sw_opts, proxy_url

def render_html(url: str, wait_selector: str = None, headless: bool = True, timeout_s: int = 45) -> str:
    """
    Headless Chrome via Selenium for JS-heavy pages or when HTTP fetch looks blocked.
    If PROXY_* env present, routes traffic through that proxy.
    """
    # Lazy import to avoid heavy deps at module import
    try:
        from seleniumwire import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
    except Exception as e:
        raise RuntimeError(f"Selenium deps missing: {e}")

    sw_opts, _ = _get_proxy_caps()

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1200,2000")
    options.add_argument("--lang=en-US")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")

    driver = webdriver.Chrome(
        executable_path=ChromeDriverManager().install(),
        options=options,
        seleniumwire_options=sw_opts or {}
    )
    try:
        driver.get(url)
        if wait_selector:
            try:
                WebDriverWait(driver, min(timeout_s, 30)).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
            except Exception:
                pass
        # small settle
        time.sleep(2)
        html = driver.page_source or ""
        return html
    finally:
        try:
            driver.quit()
        except Exception:
            pass
'''

def _shields_source() -> str:
    return r'''\
def looks_blocked(html: str) -> bool:
    if not html: return True
    low = html.lower()
    needles = [
        "captcha", "robot check", "are you a robot", "verify you are human",
        "access denied", "forbidden", "please enable cookies", "pardon the interruption",
        "temporarily blocked", "unusual traffic", "automated queries"
    ]
    return any(k in low for k in needles)
'''

def _runner_source() -> str:
    return r'''\
import os, sys, json, importlib.util

def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error":"Usage: runner.py <url> <page> <limit> <session_id>"}))
        sys.exit(1)
    url = sys.argv[1]
    page = int(sys.argv[2])
    limit = int(sys.argv[3])
    session_id = sys.argv[4]
    try:
        spec = importlib.util.spec_from_file_location("scraper", "scraper.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            print(json.dumps({"error":"No run() function found in scraper"})); return
        result = mod.run(url, page, limit, session_id)
        if isinstance(result, list):
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(json.dumps({"error": f"Invalid result type: {type(result)}"}))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))

if __name__ == "__main__":
    main()
'''

#groq client
def _llm_client():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set — required when LLM_PROVIDER=groq.")
        try:
            from groq import Groq
            return ("groq", Groq(api_key=key))
        except Exception as e:
            raise RuntimeError(f"Groq client import failed: {e}")
    else:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set — required when LLM_PROVIDER=openai.")
        try:
            from openai import OpenAI
            return ("openai", OpenAI(api_key=key))
        except Exception as e:
            raise RuntimeError(f"OpenAI client import failed: {e}")


# Prompt builders (Dynamic codegen + Fix)

GUIDANCE = """
You are generating Python **scraper code** for review extraction on an arbitrary website.

HARD CONSTRAINTS (must follow exactly):
- Imports: 
    from bd_sdk import bd_fetch_html
    from renderers import render_html as _render_html  # optional JS fallback
    from shields import looks_blocked
    from bs4 import BeautifulSoup
- Implement a SINGLE entry point:
    def run(url: str, page: int, limit: int, session_id: str) -> list[dict]
- Return: list of dicts with keys: text, rating, date, author (rating/date/author may be None)
- Respect `page`: construct/follow pagination (rel="next", aria labels, ?page=, pageNumber=, data-key ajax "load more")
- Respect `limit`: cap results per call.
- Use **bd_fetch_html** first. If content looks blocked or suspicious (looks_blocked or too short), then try **_render_html(url)** as fallback.
- Parse using BeautifulSoup("lxml").
- Robust selectors: try multiple fallbacks (common review containers + inner fields).
- Handle missing elements with try/except; NEVER raise — return [] on failure.
- Add debug prints: page URL, which path used (bd vs selenium), counts found, pagination guess.

Extraction heuristics:
- A "review" container often repeats; look for role="article", data-*="review", class contains review/testimonial/comment, or blocks with nearby stars + text.
- text: prefer long text blocks inside the container; strip whitespace.
- rating: parse floats/ints from patterns like "4.5 out of 5", "8/10", "★★★★★", etc.
- date: strings near header/footer (time, .date, [datetime], etc.)
- author: byline, profile link/name.

Pagination heuristics:
- First try: rel="next", link/button with aria-label*='Next', text 'Next'
- Then: query params (?page=, page=, pageNumber=, p=)
- Then: data-key ajax "load more" endpoints if visible in HTML (include 1 hop if trivial)
- If no obvious pagination and page>1, return [].

Output must be deterministic and within function signature.
"""

def build_generation_prompt(url: str, html_sample: str) -> str:
    return f"{GUIDANCE}\n\nTARGET URL:\n{url}\n\nSAMPLED HTML (truncated):\n<<<HTML_START>>>\n{html_sample}\n<<<HTML_END>>>"

def _extract_python_code(s: str) -> str:
    """
    Pull Python code out of a Markdown model response.
    - Prefers ```python ...``` blocks (and strips the 'python' header line).
    - Falls back to the first fenced block, else the raw string.
    """
    if not s:
        return ""
    if "```" not in s:
        t = s.lstrip()
        if t.lower().startswith("python\n"):
            return t.split("\n", 1)[1] if "\n" in t else ""
        if t.strip().lower() == "python":
            return ""
        return t

    parts = s.split("```")  # fenced blocks are at indices 1,3,5,...
    first_fenced = None
    for block in parts[1::2]:
        b = block.strip()
        if b.lower().startswith("python"):
            return b.split("\n", 1)[1] if "\n" in b else ""
        if first_fenced is None:
            first_fenced = b
    return (first_fenced or "").strip()


def call_llm_generate(url: str, html_sample: str) -> str:
    provider, client = _llm_client()
    prompt = build_generation_prompt(url, html_sample)

    model = None
    if provider == "groq":
        
        model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write robust, production-grade Python scrapers that obey constraints exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    raw = resp.choices[0].message.content or ""
    return _extract_python_code(raw)


def call_llm_fix(original_code: str, error: str, url: str, html_sample: str, error_history: List[str]) -> str:
    provider, client = _llm_client()

    model = None
    if provider == "groq":
        model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prev = "\n".join(f"- {e}" for e in error_history[-6:])
    fix_instructions = f"""
The scraper failed. FIX it and return ONLY corrected Python code for the same constraints.

URL: {url}

Last Error:
{error}

Recent Errors:
{prev}

SAMPLED HTML (truncated):
<<<HTML_START>>>
{html_sample}
<<<HTML_END>>>

Keep EXACT imports and signature, add more fallbacks for selectors, pagination, shield handling, and better debug prints.
Return [] on failure. Use _render_html fallback if bd_fetch_html looks blocked or too small.
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You fix Python scrapers precisely and obey constraints."},
            {"role": "user", "content": fix_instructions},
            {"role": "user", "content": original_code}
        ],
        temperature=0.1
    )
    raw = resp.choices[0].message.content or ""
    return _extract_python_code(raw)



# Sandbox builder + executor

def create_sandbox_environment(code_src: str, bd_cfg: BrightDataConfig) -> str:
    workdir = tempfile.mkdtemp(prefix="scraper_sbx_")
    paths = {
        "scraper":   os.path.join(workdir, "scraper.py"),
        "bd_sdk":    os.path.join(workdir, "bd_sdk.py"),
        "renderers": os.path.join(workdir, "renderers.py"),
        "shields":   os.path.join(workdir, "shields.py"),
        "runner":    os.path.join(workdir, "runner.py")
    }
    with open(paths["scraper"], "w", encoding="utf-8") as f:
        f.write(code_src)
    with open(paths["bd_sdk"], "w", encoding="utf-8") as f:
        f.write(_bd_sdk_source(bd_cfg))
    with open(paths["renderers"], "w", encoding="utf-8") as f:
        f.write(_renderers_source())
    with open(paths["shields"], "w", encoding="utf-8") as f:
        f.write(_shields_source())
    with open(paths["runner"], "w", encoding="utf-8") as f:
        f.write(_runner_source())

    # deps: requests, bs4, lxml, selenium, selenium-wire, webdriver-manager
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
             "--target", workdir,
             "requests", "beautifulsoup4", "lxml", "selenium==4.*", "selenium-wire", "webdriver-manager"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] pip failed: {e}")

    return workdir

def _read_json_from_stdout(stdout: str) -> Any:
    if not stdout:
        return {"error": "No output"}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        lines = stdout.strip().splitlines()
        for line in reversed(lines):
            t = line.strip()
            if t.startswith("{") or t.startswith("["):
                try:
                    return json.loads(t)
                except:
                    continue
        return {"error": stdout[:500]}

def execute_code_sandbox(code_src: str, url: str, session_id: str, bd_cfg: BrightDataConfig,
                         page: int = 1, limit: int = 5, timeout_s: int = 120) -> Dict[str, Any]:
    workdir = create_sandbox_environment(code_src, bd_cfg)
    try:
        env = {**os.environ, "PYTHONPATH": workdir}
        cmd = [sys.executable, os.path.join(workdir, "runner.py"), url, str(page), str(limit), session_id]
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            return {"success": False, "error": f"Process failed {proc.returncode}: {stderr or stdout}"}
        data = _read_json_from_stdout(stdout)
        if isinstance(data, dict) and "error" in data:
            return {"success": False, "error": data["error"]}
        ok = validate_review_structure(data)
        return {"success": ok, "data": data if ok else None, "error": None if ok else "Invalid output structure"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout_s}s"}
    except Exception as e:
        return {"success": False, "error": f"Sandbox failed: {e}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def validate_review_structure(data: Any) -> bool:
    if not isinstance(data, list): return False
    # Empty list allowed (signals end/blocked)
    for r in data:
        if not isinstance(r, dict): return False
        t = r.get("text")
        if not t or not isinstance(t, str) or len(t.strip()) < 3: return False
        # rating/date/author optional
    return True


# HTML sampler (HTTP first, then Selenium fallback)

def sample_html_for_prompt(url: str, session_id: str, bd_cfg: BrightDataConfig) -> str:
    workdir = tempfile.mkdtemp(prefix="bd_sample_")
    try:
        # write support modules so imports work
        with open(os.path.join(workdir, "bd_sdk.py"), "w", encoding="utf-8") as f:
            f.write(_bd_sdk_source(bd_cfg))
        with open(os.path.join(workdir, "renderers.py"), "w", encoding="utf-8") as f:
            f.write(_renderers_source())
        with open(os.path.join(workdir, "shields.py"), "w", encoding="utf-8") as f:
            f.write(_shields_source())

        sys.path.insert(0, workdir)
        from bd_sdk import bd_fetch_html  
        from renderers import render_html  
        from shields import looks_blocked  

        # Try HTTP API first
        try:
            html = bd_fetch_html(url, session_id=session_id, render=True, retries=bd_cfg.retries)
        except Exception as e:
            html = f"<!-- HTTP fetch error: {e} -->"

        if not html or len(html) < SAMPLE_HTML_MIN_LEN or looks_blocked(html):
            # JS render fallback via Selenium+proxy if available
            try:
                html2 = render_html(url, wait_selector=None, headless=True, timeout_s=45)
                if html2 and len(html2) > len(html):
                    html = html2
            except Exception as e:
                html += f"\n<!-- Selenium fallback error: {e} -->"

        return (html or "")[:SAMPLE_HTML_MAX_CHARS]
    except Exception as e:
        return f"<!-- FAILED TO SAMPLE HTML: {e} -->"
    finally:
        try:
            sys.path.remove(workdir)
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


def execute_with_retry(code_src: str, url: str, page: int, session_id: str,
                       bd_cfg: BrightDataConfig, max_retries: int = 3) -> List[Dict]:
    for attempt in range(1, max_retries + 1):
        res = execute_code_sandbox(code_src, url, session_id, bd_cfg, page=page, limit=50)
        if res["success"] and res["data"] is not None:
            return res["data"]
        err = res.get("error") or "Unknown"
        print(f"[RETRY] page={page} attempt={attempt} error={err}")
        time.sleep(min(2 ** attempt + random.random(), 12))
    raise RuntimeError(f"execute_with_retry exhausted for page {page}")

#pseudocode flow - extraction and generation
def generate_scraper_code(url: str, html_sample: str) -> str:
    return call_llm_generate(url, html_sample)

def attempt_recovery(current_code: str, url: str, page_number: int, error: str,
                     error_history: List[str], bd_cfg: BrightDataConfig, session_id: str) -> Dict[str, Any]:
    print(f"[RECOVERY] Fixing due to: {error}")
    html_sample = sample_html_for_prompt(url, session_id, bd_cfg)
    new_code = call_llm_fix(current_code, error, url, html_sample, error_history)
    smoke = execute_code_sandbox(new_code, url, session_id, bd_cfg, page=1, limit=5, timeout_s=120)
    return {"success": smoke["success"], "new_code": new_code if smoke["success"] else None, "error": smoke.get("error")}

def extract_all_reviews(working_code: str, url: str, proxy_pool, bd_cfg: BrightDataConfig,
                        error_history: List[str]) -> Tuple[List[Dict], str]:
    all_reviews: List[Dict] = []
    page_number = 1
    consecutive_failures = 0
    consecutive_empty_pages = 0
    has_more_pages = True

    print(f"[EXTRACT] start {url}")

    while has_more_pages and page_number <= MAX_PAGES:
        sess = proxy_pool.get_next()
        try:
            page_reviews = execute_with_retry(working_code, url, page_number, sess, bd_cfg, max_retries=3)
            if not page_reviews:
                consecutive_empty_pages += 1
                print(f"[EXTRACT] empty page {page_number} (streak {consecutive_empty_pages})")
                if consecutive_empty_pages > 2:
                    has_more_pages = False
                    break
            else:
                consecutive_empty_pages = 0
                all_reviews.extend(page_reviews)
                print(f"[EXTRACT] page {page_number}: +{len(page_reviews)} (total {len(all_reviews)})")

            page_number += 1
            consecutive_failures = 0
            time.sleep(random.uniform(2.0, 5.0))  # rate limit

        except Exception as e:
            consecutive_failures += 1
            msg = str(e)
            print(f"[ERROR] page {page_number} failed: {msg}")
            error_history.append(msg)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("[RECOVERY] invoking LLM fix…")
                fix = attempt_recovery(working_code, url, page_number, msg, error_history, bd_cfg, sess)
                if fix["success"] and fix["new_code"]:
                    working_code = fix["new_code"]
                    consecutive_failures = 0
                    print("[RECOVERY] new code installed; retrying same page")
                    continue
                else:
                    print("[RECOVERY] fix failed; stopping")
                    break
            else:
                time.sleep(8)

    print(f"[EXTRACT] done total={len(all_reviews)}")
    return all_reviews, working_code


def save_reviews(reviews: List[Dict], tenant_id: str):
    os.makedirs("output", exist_ok=True)
    ts = int(time.time())
    fp = os.path.join("output", f"{tenant_id}_reviews_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "tenant_id": tenant_id, "review_count": len(reviews), "reviews": reviews},
                  f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {len(reviews)} reviews -> {fp}")

#main function for generation and self-healing
def main(tenant_id: str, target_url: str, bd_cfg: BrightDataConfig) -> List[Dict]:
    print("=== Adaptive Review Extraction System (Dynamic LLM + Selenium) ===")
    print(f"[MAIN] tenant={tenant_id}")
    print(f"[MAIN] url={target_url}")

    # Initialize
    proxy_pool = initialize_brightdata_proxy(bd_cfg)
    error_history: List[str] = []

    # Ask AI to Write the Code (with LIVE HTML sample)
    print("[CODEGEN] sampling HTML for prompt…")
    sample = sample_html_for_prompt(target_url, proxy_pool.get_next(), bd_cfg)
    print(f"[CODEGEN] sample_len={len(sample)}")
    scraper_code = generate_scraper_code(target_url, sample)

    # Self-healing loop 
    working_code: Optional[str] = None
    for attempt in range(1, MAX_CODE_GENERATION_ATTEMPTS + 1):
        print(f"[CODEGEN] test attempt {attempt}/{MAX_CODE_GENERATION_ATTEMPTS}")
        test_sess = proxy_pool.get_next()
        test = execute_code_sandbox(scraper_code, target_url, test_sess, bd_cfg, page=1, limit=5, timeout_s=120)
        if test["success"]:
            working_code = scraper_code
            print(f"[CODEGEN] ✅ success with {len(test['data'])} sample items")
            break
        err = test.get("error", "unknown")
        error_history.append(err)
        print(f"[CODEGEN] ❌ fail: {err}")
        if attempt < MAX_CODE_GENERATION_ATTEMPTS:
            fix_html = sample_html_for_prompt(target_url, proxy_pool.get_next(), bd_cfg)
            scraper_code = call_llm_fix(scraper_code, err, target_url, fix_html, error_history)

    if working_code is None:
        raise RuntimeError("Failed to generate working scraper after max attempts")

    # Collect all reviews (pagination + retry + LLM recovery)
    print("[MAIN] full extraction…")
    all_reviews, final_code = extract_all_reviews(working_code, target_url, proxy_pool, bd_cfg, error_history)

    # Save in batches
    if all_reviews:
        print("[MAIN] saving…")
        for i in range(0, len(all_reviews), BATCH_SIZE):
            save_reviews(all_reviews[i:i+BATCH_SIZE], f"{tenant_id}_batch_{i//BATCH_SIZE+1}")
    else:
        print("[MAIN] no reviews found")

    print(f"[MAIN] done total={len(all_reviews)}")
    return all_reviews


if __name__ == "__main__":
    TENANT_ID = os.getenv("TENANT_ID", "demo-tenant")
    TARGET_URL = os.getenv("TARGET_URL", "https://example.com/reviews")
    BD_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
    if not BD_API_KEY:
        print("ERROR: set BRIGHTDATA_API_KEY"); sys.exit(1)

    bd_cfg = BrightDataConfig(
        api_key=BD_API_KEY,
        zone=os.getenv("BRIGHTDATA_ZONE", "web_unlocker1"),
        country=os.getenv("BRIGHTDATA_COUNTRY", "US"),
        endpoint=os.getenv("BRIGHTDATA_ENDPOINT", "https://api.brightdata.com/request"),
        render_js_hint=os.getenv("BRIGHTDATA_RENDER_JS", "true").lower() == "true",
        timeout_s=int(os.getenv("BRIGHTDATA_TIMEOUT", "40")),
        retries=3
    )

    try:
        reviews = main(TENANT_ID, TARGET_URL, bd_cfg)
        print(f"\n✅ SUCCESS total={len(reviews)}")
        if reviews:
            s = reviews[0]
            print(f"Sample: text={s.get('text','')[:120]}… rating={s.get('rating')} author={s.get('author')} date={s.get('date')}")
    except Exception as e:
        logging.exception("FAILED")
        print(f"\n❌ FAILED: {e}")
        sys.exit(1) 
