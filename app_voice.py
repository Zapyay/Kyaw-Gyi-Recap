#@title 🔊 Video အသံပြောင်း — AI အသံ / ကိုယ့်အသံ (AssemblyAI + Gemini + VoxCPM2)
# ============ ဒီ ၂ ကြောင်းကို GitHub ပေါ်မှာ တစ်ခါပဲ ပြင်ပါ ============
SERVER_URL = "https://script.google.com/macros/s/AKfycbxN-nSr2CV6f2nsClYzrTYBzaCQzeCEfCETMvvqOmwqNdEmTLa5RcQiQHG1g-6zu0kn/exec"  # Apps Script Web App URL (Admin quota server) — ဗလာထားရင် ကန့်သတ်ချက်မရှိ (စမ်းသပ်ရန်သာ)
ID_SALT = "zapyartun-recap-2026-k9x"   # တစ်ခါသတ်မှတ်ပြီးရင် မပြောင်းပါနဲ့ (ပြောင်းရင် User ID အားလုံး ပြောင်းသွားမယ်)
# ====================================================================
# API key တွေကို Colab cell ထဲမှာ ကြိုသတ်မှတ်ထားရင် ယူသုံးမယ်၊ မရှိရင် Run တဲ့အခါ မေးမယ် (GitHub ပေါ်မှာ key မထည့်ပါနဲ့)
GEMINI_API_KEY = globals().get("GEMINI_API_KEY", "") or ""
ASSEMBLYAI_API_KEY = globals().get("ASSEMBLYAI_API_KEY", "") or ""
COOKIES_FILE = globals().get("COOKIES_FILE", "") or "/content/cookies.txt"

import os, re, sys, json, time, math, glob, shutil, subprocess, threading, uuid, traceback, getpass, unicodedata, tempfile, hashlib
from concurrent.futures import ThreadPoolExecutor
if not shutil.which("nvidia-smi"): raise SystemExit("❌ GPU မရှိပါ။ Runtime → Change runtime type → T4 GPU ရွေးပြီး ပြန် Run ပါ")
KEYS_FILE = "/content/api_keys.json"
try:
    _kj = json.load(open(KEYS_FILE))
    if not GEMINI_API_KEY: GEMINI_API_KEY = _kj.get("gemini", "") or ""
    if not ASSEMBLYAI_API_KEY: ASSEMBLYAI_API_KEY = _kj.get("aai", "") or ""
except Exception: pass
import requests

# ---- User ID + ထုတ်ခွင့် (Admin server)
def srv(**p):
    if not SERVER_URL: return {"ok": True, "remaining": None}
    last = ""
    for _ in range(3):
        try: return requests.get(SERVER_URL, params=p, timeout=30).json()
        except Exception as e: last = str(e)[:100]; time.sleep(1.5)
    return {"ok": False, "msg": "Server ချိတ်မရပါ: " + last}

def make_uid():
    email = ""
    try:
        from google.colab import auth
        auth.authenticate_user()
        out = subprocess.run(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], capture_output=True, text=True).stdout.strip()
        email = (out.splitlines() or [""])[0].strip().lower()
        if "@" not in email:
            tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
            email = str(requests.get("https://www.googleapis.com/oauth2/v2/userinfo", headers={"Authorization": "Bearer " + tok}, timeout=20).json().get("email", "")).lower()
    except Exception as e:
        print("Google account ဖတ်မရပါ:", str(e)[:120])
    if "@" in email:
        return "U-" + hashlib.sha256(f"{ID_SALT}|{email}".encode()).hexdigest()[:8].upper()
    fp = "/content/.uid"
    try:
        if os.path.exists(fp): return open(fp).read().strip()
    except Exception: pass
    u = "U-" + uuid.uuid4().hex[:8].upper()
    try: open(fp, "w").write(u)
    except Exception: pass
    print("⚠️ Google account မရလို့ ID ကို ယာယီထုတ်ထားတယ် (Runtime ပြန်စရင် ပြောင်းနိုင်ပါတယ်)")
    return u

UID = make_uid() if SERVER_URL else "U-TEST0000"
_q = srv(action="check", id=UID)
print("🆔 User ID:", UID, "| ထုတ်ခွင့်ကျန်:", (_q.get("remaining") if (SERVER_URL and _q.get("ok")) else "ကန့်သတ်မထား"))
if SERVER_URL and not _q.get("ok"): print("⚠️", _q.get("msg"))

import importlib.metadata as md
pins = []
for n in ["numpy", "torch", "torchaudio", "torchvision", "scipy", "numba", "pandas", "scikit-learn"]:
    try: pins.append(f"{n}=={md.version(n)}")
    except Exception: pass
open("constraints.txt", "w").write("\n".join(pins))
print("🔧 Package တပ်ဆင်နေပါတယ်…")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "constraints.txt", "voxcpm", "soundfile", "fastapi", "uvicorn", "python-multipart", "google-genai", "requests"], capture_output=True, text=True)
if r.returncode: print(r.stderr[-2000:]); raise SystemExit("❌ Package တပ်ဆင်မှု မအောင်မြင်ပါ")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp[default]", "curl_cffi"], capture_output=True, text=True)
# မြန်မာ စာတန်းကို video ပေါ်ရိုက်ထည့်ဖို့ font
import numpy as np, soundfile as sf, uvicorn
from voxcpm import VoxCPM
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
print("⏳ VoxCPM2 load နေပါတယ်…")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = getattr(model.tts_model, "sample_rate", 48000)
gc = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]
def discover_models():
    if gc is None: return []
    try:
        names = []
        for m in gc.models.list():
            n = (getattr(m, "name", "") or "").replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if n.startswith("gemini") and "flash" in n and (not acts or "generateContent" in acts) and not any(x in n for x in ("image", "tts", "live", "audio", "embed", "exp", "robotics", "computer", "native", "thinking")):
                names.append(n)
        def key(n):
            v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return (-(float(v.group(1)) if v else 0), "lite" in n, "preview" in n, n)
        return sorted(set(names), key=key)[:8]
    except Exception as e:
        print("model list မရပါ:", str(e)[:120]); return []
_d = discover_models()
if _d: MODELS = _d + [m for m in ("gemini-flash-latest",) if m not in _d]
print("🤖 Gemini models:", MODELS[:5])
DEAD, COOL = set(), {}
AAI = "https://api.assemblyai.com/v2"
WORK, FPS = "/content/recap", 30
GAP = 8.0
MIN_SPEED = 0.4
LINE_MAX = 26
MAX_MIN = 15   # video အရှည်ဆုံး (မိနစ်)
# scene ခွဲပုံ — စကားတစ်ကြောင်းချင်း အချိန်မှန်အောင် scene တိုတိုခွဲမယ်
MERGE_SPAN, MERGE_MAX, MERGE_GAP = 3.5, 7.0, 1.0
os.makedirs(WORK, exist_ok=True)
JOBS, GPU = {}, threading.RLock()
DL = {}   # link ဒေါင်းတဲ့ အလုပ်တွေ

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError((r.stderr or "")[-400:])
    return r.stdout

def sh_prog(cmd, total, cb):
    ef = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:], stdout=subprocess.PIPE, stderr=ef, text=True)
    for line in p.stdout:
        if line.startswith("out_time_ms="):
            try: cb(min(.99, int(line.split("=")[1]) / 1e6 / max(1.0, total)))
            except Exception: pass
    p.wait(); ef.seek(0); err = ef.read().decode(errors="ignore"); ef.close()
    if p.returncode: raise RuntimeError(err[-400:])

def probe(p):
    j = json.loads(sh(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", p]))
    v = [s for s in j["streams"] if s["codec_type"] == "video"][0]
    w, h = int(v["width"]), int(v["height"])
    rot = int(v.get("tags", {}).get("rotate", 0) or 0)
    for sd in v.get("side_data_list", []):
        if "rotation" in sd: rot = int(sd["rotation"])
    if abs(rot) in (90, 270): w, h = h, w
    return float(j["format"]["duration"]), w, h

def mmss(x): return f"{int(x // 60):02d}:{int(x % 60):02d}"

def split(text, n=240):
    out, cur = [], ""
    for p in re.split(r"(?<=[။?!.\n])", text):
        if len(cur) + len(p) > n and cur: out.append(cur.strip()); cur = ""
        if len(p) > n: out += [p[i:i+n].strip() for i in range(0, len(p), n)]
        else: cur += p
    if cur.strip(): out.append(cur.strip())
    return [x for x in out if x]

def jparse(t): return json.loads(re.sub(r"^```(?:json)?|```$", "", t.strip()).strip())

TH = {}
def _think_variants():
    v = []
    try: v.append(types.ThinkingConfig(thinking_budget=0))
    except Exception: pass
    try: v.append(types.ThinkingConfig(thinking_level="low"))
    except Exception: pass
    v.append(None)
    return v
THV = _think_variants()

def gcall(contents, temp=0.3, tmax=420):
    # 429 (quota) / 503 (high demand) ဖြစ်ရင် model ကို ခဏ ရပ်ထားပြီး၊ ပြန်ကောင်းတဲ့အထိ စောင့်ပြီး ထပ်စမ်းမယ်
    if gc is None: raise RuntimeError("Gemini API key မထည့်ရသေးပါ — Website အပေါ်ညာထောင့် ⚙️ Settings မှာ ထည့်ပါ")
    last, errs, t_start = "", {}, time.time()
    while True:
        for m in MODELS:
            if m in DEAD or COOL.get(m, 0) > time.time(): continue
            for vi in range(TH.get(m, 0), len(THV)):
                try:
                    kw = {"thinking_config": THV[vi]} if THV[vi] is not None else {}
                    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=temp, **kw)
                    txt = gc.models.generate_content(model=m, contents=contents, config=cfg).text
                    TH[m] = vi
                    return txt
                except Exception as e:
                    s = str(e); last = f"{m}: {s[:250]}"; errs[m] = s[:110]
                    if any(x in s for x in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED")):
                        raise RuntimeError("Gemini API key မှားနေ/ခွင့်မပြုပါ: " + s[:250])
                    if "404" in s or "NOT_FOUND" in s: DEAD.add(m); break
                    if "thinking" in s.lower() and vi < len(THV) - 1: continue
                    if "429" in s or "RESOURCE_EXHAUSTED" in s:
                        mm = re.search(r"retry in ([\d.]+)s", s) or re.search(r"retryDelay['\"]?: ?['\"]?(\d+)", s)
                        wait = float(mm.group(1)) + 3 if mm else 60.0
                        if "PerDay" in s or "per day" in s.lower(): wait = 3600.0
                        COOL[m] = time.time() + min(wait, 3600.0)
                    else: COOL[m] = time.time() + 15
                    break
        live = [m for m in MODELS if m not in DEAD]
        if not live: break
        nxt = min(COOL.get(m, 0) for m in live) - time.time()
        if time.time() - t_start > tmax or nxt > 300: break
        time.sleep(max(2.0, min(30.0, nxt)))
    raise RuntimeError("Gemini ခေါ်လို့မရပါ — " + (" | ".join(f"{k}: {v}" for k, v in list(errs.items())[:4]) or last) +
                       "\n💡 Gemini quota ပြည့်နေ/လူများနေပါတယ် — ၁–၂ မိနစ်စောင့်ပြီး History ကနေ ပြန်လုပ်ပါ။ မကြာခဏဖြစ်ရင် Google AI Studio မှာ Billing ဖွင့်ထားတဲ့ API key (သို့) project အသစ်က key အသစ် သုံးပါ")

if GEMINI_API_KEY:
    try:
        gcall(["Reply with JSON array [1]"], tmax=45)
        print("✅ Gemini OK:", [m for m in MODELS if m not in DEAD and COOL.get(m, 0) <= time.time()][:1])
    except Exception as e:
        print("⚠️ Gemini key စစ်မရပါ — Website ⚙️ Settings မှာ key ပြန်စစ်ပါ:", str(e)[:200])
else:
    print("ℹ️ API key မထည့်ရသေးပါ — Website အပေါ်ညာထောင့် ⚙️ မှာ ထည့်ပြီး Save နှိပ်ပါ")

# ---- မြန်မာစာ helper
def clusters(t):
    out = []
    for i, ch in enumerate(t):
        nx = t[i + 1] if i + 1 < len(t) else ""
        if out and (unicodedata.category(ch).startswith("M") or out[-1].endswith("\u1039") or nx == "\u103a"): out[-1] += ch
        else: out.append(ch)
    return out

def fix_my(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub("[\u200b\u200c\u200d\u2060\ufeff*_#`]", "", t)
    t = t.replace("\u1039\u1039", "\u1039").replace("\u103a\u103a", "\u103a")
    t = re.sub(r"([\u102b-\u103e])\1+", r"\1", re.sub(r"[ \t]+", " ", t))
    t = re.sub(r"\s+([၊။])", r"\1", t)
    return t.strip()


# ---- Link ကနေ video ဒေါင်း (TikTok / YouTube / Facebook / Rednote …)
def tikwm_dl(url, base, prog):
    j = requests.get("https://www.tikwm.com/api/", params={"url": url, "hd": 1}, timeout=40, headers={"User-Agent": "Mozilla/5.0"}).json()
    dt = j.get("data") or {}
    pu = dt.get("hdplay") or dt.get("play")
    if not pu: raise RuntimeError(str(j.get("msg"))[:100])
    if pu.startswith("/"): pu = "https://www.tikwm.com" + pu
    path = base + ".mp4"
    with requests.get(pu, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        tb = int(r.headers.get("content-length", 0) or 0); got = 0
        with open(path, "wb") as f:
            for ch in r.iter_content(1 << 16):
                f.write(ch); got += len(ch)
                if tb: prog(min(.95, got / tb), f"{got / 1e6:.0f}/{tb / 1e6:.0f} MB")
    return path, (dt.get("title") or "tiktok").strip()[:100]

def fetch_video(url, jid, prog):
    import yt_dlp
    base = os.path.join(WORK, "dl_" + jid)
    def hook(dd):
        if dd.get("status") == "downloading":
            tb = dd.get("total_bytes") or dd.get("total_bytes_estimate") or 0
            got = dd.get("downloaded_bytes", 0) or 0
            if tb: prog(min(.95, got / tb), f"{got / 1e6:.0f}/{tb / 1e6:.0f} MB")
            else: prog(.5, f"{got / 1e6:.0f} MB")
    common = {"merge_output_format": "mp4", "outtmpl": base + ".%(ext)s", "noplaylist": True, "quiet": True, "no_warnings": True,
              "progress_hooks": [hook], "retries": 5, "fragment_retries": 5, "socket_timeout": 30, "concurrent_fragment_downloads": 4,
              "js_runtimes": {"node": {}}}
    if os.path.exists(COOKIES_FILE): common["cookiefile"] = COOKIES_FILE
    FMT = "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"
    variants = [{"format": FMT}]
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        variants.append({"format": FMT, "impersonate": ImpersonateTarget("chrome")})
    except Exception: pass
    variants += [{"format": FMT, "extractor_args": {"youtube": {"player_client": ["tv", "web_safari"]}}},
                 {"format": FMT, "extractor_args": {"youtube": {"player_client": ["android_vr", "ios"]}}},
                 {"format": "b[height<=720]/b"}]
    info, last, files = None, "", []
    for v in variants:
        for f in glob.glob(base + ".*"):
            try: os.remove(f)
            except Exception: pass
        try:
            with yt_dlp.YoutubeDL({**common, **v}) as y:
                info = y.extract_info(url, download=True)
            files = [f for f in glob.glob(base + ".*") if not f.endswith((".part", ".ytdl", ".json"))]
            if files: break
            info = None
        except Exception as e:
            last = re.sub(r"\x1b\[[0-9;]*m", "", str(e))[-300:]; info = None
    if not info and re.search(r"tiktok\.com|douyin\.com", url):
        try: return tikwm_dl(url, base, prog)
        except Exception as e: last += " | tikwm: " + str(e)[:100]
    if not info:
        raise RuntimeError("Link ကနေ video ဒေါင်းမရပါ (နည်းလမ်းအားလုံး စမ်းပြီး): " + last + "\n💡 YouTube/Facebook ဆိုရင် cookies.txt ကို /content/cookies.txt မှာ တင်ပြီး ပြန် Run ပါ၊ (သို့) video ကို ဖိုင်အဖြစ် တင်ပါ")
    ents = info.get("entries") if isinstance(info, dict) else None
    if ents: info = next((x for x in ents if x), info)
    files.sort(key=os.path.getsize, reverse=True)
    return files[0], (info.get("title") or "video").strip()

def dl_worker(did, url):
    d = DL[did]
    try:
        path, title = fetch_video(url, did, lambda f, s: d.update(f=f, d=s))
        d.update(state="done", path=path, title=title, f=1.0, d="")
    except Exception as e:
        d.update(state="err", msg=str(e)[-400:])

# ---- AssemblyAI: video ထဲက စကားကို စာထုတ် (Dubbing မှာ စကားပြောသူ ခွဲ)
def aai_transcribe(audio_path, lang, prog, diar=False):
    H = {"authorization": ASSEMBLYAI_API_KEY}
    prog(.1, "အသံ တင်နေ")
    with open(audio_path, "rb") as f:
        r = requests.post(f"{AAI}/upload", headers=H, data=f, timeout=900)
    if r.status_code == 401: raise RuntimeError("AssemblyAI API key မှားနေပါတယ်")
    r.raise_for_status()
    url = r.json()["upload_url"]
    base = {"audio_url": url}
    if lang and lang != "auto": base["language_code"] = lang
    else: base["language_detection"] = True
    if diar: base["speaker_labels"] = True
    tries = [base, {**base, "speech_models": ["universal-3-pro", "universal-2"]}, {**base, "speech_model": "best"}, {**{k: v for k, v in base.items() if k != "language_code"}, "language_detection": True}]
    err = ""
    for body in tries:
        r = requests.post(f"{AAI}/transcript", headers={**H, "content-type": "application/json"}, json=body, timeout=60)
        if r.status_code == 200: break
        err = r.text
    else: raise RuntimeError("AssemblyAI: " + err[:300])
    tid = r.json()["id"]
    t0 = time.time()
    while True:
        time.sleep(3)
        j = requests.get(f"{AAI}/transcript/{tid}", headers=H, timeout=60).json()
        if j.get("status") == "completed": break
        if j.get("status") == "error": raise RuntimeError("AssemblyAI: " + str(j.get("error"))[:300])
        prog(min(.9, .3 + (time.time() - t0) / 300), "စာထုတ်နေ")
    s = requests.get(f"{AAI}/transcript/{tid}/sentences", headers=H, timeout=60).json().get("sentences", [])
    order, out = {}, []
    for x in s:
        t = str(x.get("text", "")).strip()
        if not t: continue
        k = 0
        if diar:
            lab = x.get("speaker") or "A"
            if lab not in order: order[lab] = len(order)
            k = order[lab] % 2      # စကားပြောသူ ၁ → Voice1 ၊ ၂ → Voice2
        out.append((x["start"] / 1000.0, x["end"] / 1000.0, t, k))
    return out, j.get("language_code", "")

# ---- Scene ဖွဲ့
# အသံနဲ့ video ကိုက်အောင်: scene တစ်ခုချင်းစီက မူရင်းစကား စတဲ့အချိန်မှာပဲ စတယ်၊
# ဒါကြောင့် ဘာသာပြန်အသံ ဟာ အဲ့ဒီစကားပြောတဲ့ အပိုင်းမှာပဲ စပြောမယ် (ကြိုမပြောတော့ဘူး)
def build_scenes(sents, dur):
    sents = sorted([(max(0.0, min(a, dur)), max(0.0, min(b, dur)), t, k) for a, b, t, k in sents if t], key=lambda x: x[0])
    groups, cur = [], None
    for a, b, t, k in sents:
        if cur and cur[3] == k and (cur[1] - cur[0] < MERGE_SPAN) and (b - cur[0] <= MERGE_MAX) and (a - cur[1] < MERGE_GAP):
            cur[1] = max(cur[1], b); cur[2].append(t)
        else:
            if cur: groups.append(cur)
            cur = [a, b, [t], k]
    if cur: groups.append(cur)
    sc = lambda a, b, t, k=0, sd=0.0: {"a": a, "b": b, "src": t, "t": "", "spk": k, "sd": sd}
    out, pos = [], 0.0
    for gi, (s, e, tl, k) in enumerate(groups):
        if s - pos > 0.3: out.append(sc(pos, s, "")); pos = s
        nxt = groups[gi + 1][0] if gi + 1 < len(groups) else None
        txt, sd = " ".join(tl), max(0.5, e - s)
        if nxt is None:
            if dur - e > GAP: out.append(sc(pos, e + .5, txt, k, sd)); out.append(sc(e + .5, dur, "")); pos = dur
            else: out.append(sc(pos, dur, txt, k, sd)); pos = dur
        elif nxt - e > GAP:
            out.append(sc(pos, e + .5, txt, k, sd)); out.append(sc(e + .5, nxt, "")); pos = nxt
        else:
            out.append(sc(pos, nxt, txt, k, sd)); pos = nxt
    return [x for x in out if x["src"] or x["b"] - x["a"] > 0.3]


# ---- Gemini ဘာသာပြန်
TR = """သင်ဟာ မြန်မာစာ ကျွမ်းကျင်တဲ့ video dubbing ဘာသာပြန်သူပါ။ အောက်က JSON array က video ထဲက စကားပြောတွေ (မူရင်းဘာသာစကား ဘာဖြစ်ဖြစ်) ကို အစီအစဉ်အတိုင်း ပေးထားတာပါ။ တစ်ခုချင်းကို မြန်မာလို ဘာသာပြန်ပြီး အသံထွက်ဖတ်ဖို့ သင့်တော်အောင် ရေးပါ။
- string array တစ်ခုတည်း ပြန်ပါ။ အရေအတွက်၊ အစီအစဉ် အတိအကျ တူရမယ်။
- item တစ်ခုချင်းစီရဲ့ "chars" က ရေးရမယ့် မြန်မာစာလုံး (character) အရေအတွက် ပစ်မှတ်။ ±15% ထဲ ကိုက်ပါစေ၊ ကျော်လို့မရ (အသံကို မြန်မြန်ဖတ်ခိုင်းမှာ မဟုတ်လို့ ကျော်ရင် အချိန်ပိုကြာသွားမယ်)။ ပိုရှည်နေရင် အသေးစိတ်ကို ချုံ့ပါ၊ တိုနေရင် မူရင်းအဓိပ္ပာယ်ကို သဘာဝကျကျ ရှင်းပြပါ (မူရင်းမှာ မပါတာ မဖန်တီးနဲ့)။
- စာလုံးပေါင်း တိတိကျကျ မှန်ရမယ်။ စာလုံးအသစ်ဖန်တီးခြင်း၊ မသေချာတဲ့ ရှားပါးစကားလုံး မသုံးနဲ့။ ရိုးရိုး နေ့စဉ်ပြောတဲ့ စကားလုံးတွေသာ သုံးပါ။ ဝါကျတစ်ကြောင်းချင်း ပြည့်စုံပြီး သဒ္ဒါမှန်ရမယ်။
- Unicode မြန်မာစာသာ (Zawgyi မသုံးနဲ့)။ ဂဏန်း/English ကို မြန်မာစာလုံးနဲ့ ရေးပါ။ စာကြောင်းအဆုံးမှာ ။ ထည့်ပါ။ ဇာတ်ကောင်နာမည် တစ်ပုံစံတည်း ရေးပါ။
CTX
INPUT:
"""
RECAP_HINT = "- ဒီ video ကို ဇာတ်လမ်းအကျဉ်းပြောပြတဲ့ narrator တစ်ယောက်တည်း အသံနဲ့ ဖတ်မှာပါ။ ဇာတ်ကြောင်းပြောပြသလို တစ်ဆက်တည်း သဘာဝကျကျ ရေးပါ။ item တစ်ခုချင်းစီက video ရဲ့ အဲ့ဒီအချိန်မှာ ဖြစ်နေတာကိုသာ ပြောပါ (နောက်ပိုင်း ဖြစ်မယ့်အကြောင်း ကြိုမပြောနဲ့)။"
DUB_HINT = "- ဒီ video မှာ လူနှစ်ယောက် စကားပြောနေတာပါ ('spk' က စကားပြောသူ နံပါတ်)။ အပြန်အလှန် စကားပြောသလို စကားပြော ပုံစံနဲ့ ရေးပါ။"
SHORT = "အောက်က မြန်မာစာကို အဓိပ္ပာယ်မပျက်ဘဲ {N} စာလုံးခန့်အထိ ချုံ့ပါ။ ရိုးရိုးစကားလုံး၊ စာလုံးပေါင်းမှန်၊ Unicode၊ အဆုံးမှာ ။ ထည့်ပါ။ string တစ်ခုပါတဲ့ JSON array ပြန်ပါ။\n"

LONGP = "အောက်က မြန်မာစာ narration ကို အဓိပ္ပာယ်မပျက်၊ ဇာတ်ကြောင်းနဲ့ ဆက်စပ်အောင် {N} စာလုံးခန့်အထိ ပိုရှည်အောင် ဖြည့်ရေးပါ (မမြင်ရတာ မဖန်တီးနဲ့၊ အသေးစိတ် ရှင်းပြတာ၊ ခံစားချက် ဖြည့်တာသာ)။ ရိုးရိုးစကားလုံး၊ စာလုံးပေါင်းမှန်၊ Unicode၊ အဆုံးမှာ ။ ထည့်ပါ။ string တစ်ခုပါတဲ့ JSON array ပြန်ပါ။\n"

def as_list(data):
    if isinstance(data, dict): data = next((v for v in data.values() if isinstance(v, list)), [])
    return [(x.get("text", "") if isinstance(x, dict) else str(x)) for x in data]

def translate_batch(items, hint):
    p = TR.replace("CTX", hint) + json.dumps(items, ensure_ascii=False)
    for _ in range(2):
        try:
            out = as_list(jparse(gcall([p])))
            if len(out) == len(items) and all(x.strip() for x in out): return [fix_my(x) for x in out]
        except Exception as e:
            if "Gemini ခေါ်လို့မရပါ" in str(e) or "API key" in str(e): raise
    res = []
    for it in items:
        q = TR.replace("CTX", hint) + json.dumps([it], ensure_ascii=False)
        try: o = as_list(jparse(gcall([q])))
        except Exception: o = []
        res.append(fix_my(o[0]) if o and o[0].strip() else "")
    return res

def shorten(text, n):
    o = as_list(jparse(gcall([SHORT.replace("{N}", str(int(n))) + json.dumps([text], ensure_ascii=False)], temp=0.2)))
    return fix_my(o[0]) if o and o[0].strip() else text


def lengthen(text, n):
    o = as_list(jparse(gcall([LONGP.replace("{N}", str(int(n))) + json.dumps([text], ensure_ascii=False)], temp=0.4)))
    return fix_my(o[0]) if o and o[0].strip() else text


# ---- Job state
WT = [2, 1, 8, 3, 10, 32, 30, 14, 0]
def pct(j):
    if j["done"]: return 100
    p = 0.0
    for w, x in zip(WT, j["steps"]):
        if x["s"] == "done": p += w
        elif x["s"] == "run": p += w * (x.get("f") or 0)
    return min(99, int(round(p)))

def P(job, i, state, d="", f=0.0):
    job["steps"][i] = {"s": state, "d": d, "f": f}; job["upd"] = time.time()

def refund_job(job):
    t = job.pop("ticket", None)
    if t: srv(action="refund", id=UID, ticket=t)

CAL = "မင်္ဂလာပါ။ ဒီနေ့ ရာသီဥတု အရမ်းကောင်းပါတယ်။ ကျွန်တော်တို့ အတူတူ ဇာတ်ကားကြည့်ကြရအောင်နော်။"

def tempo(au, spd, d=None):
    # အသံမြန်နှုန်း ချိန်ညှိ (အသံအနိမ့်အမြင့် မပြောင်းဘဲ)
    if abs(spd - 1.0) < 0.02 or len(au) < 200: return au
    base = os.path.join(d or WORK, "t_" + uuid.uuid4().hex[:8])
    try:
        sf.write(base + "_i.wav", au, SR)
        sh(["ffmpeg", "-y", "-i", base + "_i.wav", "-filter:a", f"atempo={max(0.5, min(2.0, spd)):.3f}", base + "_o.wav"])
        o, _ = sf.read(base + "_o.wav", dtype="float32")
        return (o if o.ndim == 1 else o.mean(axis=1)).astype(np.float32)
    finally:
        for z in ("_i.wav", "_o.wav"):
            try: os.remove(base + z)
            except Exception: pass

def ncount(t): return len(re.sub(r"[\s၊။,.!?]", "", t or ""))

def speech_secs(au):
    # စကားပြောတဲ့ အချိန်သက်သက် (အစ/အဆုံး တိတ်ဆိတ်တာ ဖယ်)
    if au is None or len(au) < 10: return 0.0
    a = np.abs(au); m = float(a.max())
    if m < 1e-4: return 0.0
    idx = np.where(a > m * 0.02)[0]
    return float(idx[-1] - idx[0]) / SR if len(idx) else 0.0

def norm_rate(au, t, target, d=None):
    # scene တိုင်း စကားပြောမြန်နှုန်း တူအောင် — အရမ်းမြန်/အရမ်းနှေးတဲ့ scene ကိုပဲ ပြင် (±၁၀% အတွင်း မထိ)
    n, sp = ncount(t), speech_secs(au)
    if n < 6 or sp < 0.8 or target <= 0: return au
    f = (n / sp) / target                    # >1 = ပုံမှန်ထက် မြန်နေ
    if 0.9 <= f <= 1.1: return au
    return tempo(au, max(0.7, min(1.4, 1.0 / f)), d)

PR, RATE0 = [], [None]      # စကားပြောမြန်နှုန်း မှတ်တမ်း (လုံး/စက္ကန့်) — အပိုင်းတိုင်း တူအောင် ညှိဖို့

def piece_norm(p, t, d=None):
    # စာကြောင်း/အပိုင်း တစ်ခုချင်းစီရဲ့ မြန်နှုန်းကို တိုင်းပြီး ပျမ်းမျှနဲ့ တူအောင် ညှိ (scene တစ်ခုအတွင်းမှာတောင် မြန်လာတာ ပြင်ပေးမယ်)
    n, sp = ncount(t), speech_secs(p)
    if n < 8 or sp < 0.7: return p
    r = n / sp
    base = RATE0[0]
    refs_ = PR[-80:] + ([base] * 3 if base else [])
    PR.append(r)
    if len(refs_) < 3: return p
    f = r / float(np.median(refs_))                # >1 = ပုံမှန်ထက် မြန်နေ
    if 0.93 <= f <= 1.07: return p
    return tempo(p, max(0.8, min(1.35, 1.0 / f)), d)

def synth(t, ref, steps=6, spd=1.0, d=None):
    ws = []
    for piece in split(t):
        w = model.generate(text=piece, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=steps)
        w = piece_norm(np.asarray(w, dtype=np.float32).reshape(-1), piece, d)
        ws += [w, np.zeros(int(SR * 0.08), np.float32)]
    au = np.concatenate(ws) if ws else np.zeros(int(SR * 0.2), np.float32)
    return tempo(au, spd, d)


def plan_scene(a, b, has_t, Lau, spd, dur):
    # Scene တိုင်း ကိုယ့်ရဲ့ မူရင်းအချိန် (a) ကနေ စပြီး ဖွင့် → ပြောသံနဲ့ ပုံ အမြဲကိုက်။ ကွာသွားတာ စုမလာအောင်။
    # အသံက scene ထက် ရှည်ရင် နည်းနည်းပဲ နှေးပေး (x0.85 အထိ)၊ ကျန်တာကို နောက်ပုံက ဆက်ယူ — ရပ်မနေစေရ။
    # return (skip, ss, dd, s, need, newcur)
    T = b - a
    if has_t and spd > 1.02 and Lau + 0.15 < T:          # အသံမြန်ရင် Video ကိုလည်း မြန်ပေး (အရင်အတိုင်း)
        dd = max(0.5, Lau + 0.15); s = min(2.5, T / dd)
        ss = min(a, max(0.0, dur - 0.5))
        return (False, ss, dd, s, max(0.3, min(dd * s, dur - ss)), 0.0)
    dd = max(T, Lau + 0.15) if has_t else T
    s = 1.0 if dd <= T + 0.05 else max(0.5 if spd < 0.98 else 0.85, T / dd)   # အသံနှေးအောင် ချိန်ထားရင် ပုံကိုလည်း အသံအတိုင်း နှေး
    use = dd * s
    ss = a
    if ss + use > dur: ss = max(0.0, dur - use)         # ပုံကုန်တော့မယ်ဆိုရင် အရင်ပုံကနေ ယူ၊ ရပ်မနေအောင်
    ss = min(ss, max(0.0, dur - 0.5))
    return (False, ss, dd, s, max(0.3, min(use, dur - ss)), 0.0)

def plan_all(scn, has_t, Ls, spd, dur):
    return [plan_scene(x["a"], x["b"], has_t[i], Ls[i], spd, dur) for i, x in enumerate(scn)]


# ---- AI အသံ ၆ ခု (VoxCPM2 voice design) — တစ်ခုနဲ့တစ်ခု မတူအောင် အသက်/အသံအနိမ့်အမြင့် ခွဲဖော်ပြ၊ တစ်ခါထုတ်ပြီး မှတ်ထားမယ် (အလုပ်တိုင်း အသံတစ်သမတ်တည်း)
AI_VOICES = {
    "သီဟ": ("ကျား", "A mature man in his 40s, deep warm baritone voice, calm and steady"),
    "မောင်မောင်": ("ကျား", "A young man in his 20s, bright energetic voice, cheerful and lively"),
    "ဇော်ကြီး": ("ကျား", "An old man in his 60s, rough raspy gravelly voice, slow and serious"),
    "နီလာ": ("မ", "A young woman, gentle and sweet soft voice"),
    "ခင်ခင်": ("မ", "A middle-aged woman, warm motherly clear voice, steady and kind"),
    "မိနီ": ("မ", "A teenage girl, high-pitched playful cute voice, lively"),
}
VDIR = "/content/voices"; os.makedirs(VDIR, exist_ok=True)
def ensure_ai_voice(name):
    p = os.path.join(VDIR, f"v{list(AI_VOICES).index(name)}.wav")
    if os.path.exists(p): return p
    w = model.generate(text=f"({AI_VOICES[name][1]}){CAL}", cfg_value=2.0, inference_timesteps=10)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    sf.write(p, w[: int(SR * 20)], SR)
    return p

# ---- AssemblyAI မရရင် Gemini နဲ့ စာထုတ် (အရန်)
GTR = """ဒီအသံထဲက စကားပြောတွေကို အချိန်နဲ့တကွ စာထုတ်ပါ (ဘာသာမပြန်ပါနဲ့၊ ပြောတဲ့ဘာသာအတိုင်း)။
JSON array သာ ပြန်ပါ: [{"start": စက္ကန့်, "end": စက္ကန့်, "text": "ပြောတဲ့ စကားအတိုင်း"}]
- မြန်မာစကားဆိုရင် Unicode မြန်မာစာနဲ့ စာလုံးပေါင်းမှန်အောင် ရေးပါ။
- start/end က အစီအစဉ်အတိုင်း၊ မထပ်စေနဲ့၊ တစ်ကြောင်း ၁၀ စက္ကန့်ထက် မရှည်စေနဲ့။"""
def gemini_transcribe(path, dur, prog):
    if gc is None: raise RuntimeError("Gemini API key မထည့်ရသေးပါ")
    prog(.1, "Gemini ဆီ အသံတင်နေ")
    f = gc.files.upload(file=path); t0 = time.time()
    while "PROCESSING" in str(getattr(f, "state", "")).upper():
        if time.time() - t0 > 300: raise RuntimeError("Gemini file ကြာလွန်းနေ")
        time.sleep(3); f = gc.files.get(name=f.name)
    try:
        prog(.5, "Gemini နဲ့ စာထုတ်နေ")
        data = jparse(gcall([f, GTR], temp=0.2))
    finally:
        try: gc.files.delete(name=f.name)
        except Exception: pass
    if isinstance(data, dict): data = next((v for v in data.values() if isinstance(v, list)), [])
    out = []
    for x in data:
        try:
            a, b, t = float(x["start"]), float(x["end"]), str(x["text"]).strip()
            if t and b > a: out.append((max(0.0, a), min(dur, b), t, 0))
        except Exception: pass
    return out

# ---- မြန်မာ စကားပါတဲ့ Video: စာလုံးပေါင်းပဲ ပြင် (စကားလုံး/အရှည် မပြောင်း)
FIXP = """သင်ဟာ မြန်မာစာ proofreader ပါ။ အောက်က JSON array ထဲက မြန်မာစာတွေဟာ အသံကနေ စက်နဲ့ စာထုတ်ထားတာမို့ စာလုံးပေါင်း/ပုဒ်မှတ် မှားနိုင်ပါတယ်။
- string array တစ်ခုတည်း ပြန်ပါ။ အရေအတွက်၊ အစီအစဉ် အတိအကျ တူရမယ်။
- ပြောတဲ့ စကားလုံးတွေ၊ အဓိပ္ပာယ်၊ အရှည်ကို မပြောင်းပါနဲ့၊ မဖြည့်၊ မဖြုတ်ပါနဲ့။ စာလုံးပေါင်းမှန်အောင်၊ ပုဒ်မှတ် (၊ ။) သင့်တော်အောင်သာ ပြင်ပါ။
- Unicode မြန်မာစာသာ။ ဂဏန်း/English ကို မြန်မာစာလုံးနဲ့ ရေးပါ။ စာကြောင်းအဆုံးမှာ ။ ထည့်ပါ။
INPUT:
"""
VC_HINT = "- ဒီ video ရဲ့ စကားပြောကို မြန်မာလို အသံထွက်ဖတ်ဖို့ ဘာသာပြန်တာပါ။ မူရင်းအဓိပ္ပာယ်အတိုင်း၊ မူရင်းမှာ မပါတာ မဖန်တီးဘဲ သဘာဝကျကျ ရေးပါ။"
def proofread(items):
    for _ in range(2):
        try:
            out = as_list(jparse(gcall([FIXP + json.dumps(items, ensure_ascii=False)], temp=0.1)))
            if len(out) == len(items) and all(x.strip() for x in out): return [fix_my(x) for x in out]
        except Exception as e:
            print("စာပြင် မရ၊ မူရင်းစာအတိုင်း သုံးမယ်:", str(e)[:120]); break
    return [fix_my(x) for x in items]

def run_job(jid, vid, dlid, voice, lang, steps, style, spd=1.0):
    job = JOBS[jid]
    d = os.path.join(WORK, jid); os.makedirs(d, exist_ok=True)
    ai = voice.startswith("ai:")
    src = os.path.join(WORK, "up_" + vid) if vid else None
    src0 = None
    vsrc = None if ai else os.path.join(WORK, "up_" + voice)
    try:
        try: stl = json.loads(style)
        except Exception: stl = {}
        # 1) Link ဒေါင်းပြီးသား / upload ပြီးသား ဖိုင်
        if dlid:
            P(job, 0, "run", "Link ဒေါင်းနေ", .02)
            while True:
                dd = DL.get(dlid)
                if not dd: raise RuntimeError("ဒေါင်းထားတဲ့ video မတွေ့ပါ")
                if dd["state"] == "err": raise RuntimeError(dd["msg"])
                if dd["state"] == "done": break
                P(job, 0, "run", dd.get("d", ""), dd.get("f", 0)); time.sleep(1)
            src = dd["path"]; job["name"] = (dd.get("title") or "video")[:120]
        P(job, 0, "done", "", 1); P(job, 1, "run")
        src0 = src
        # ✂ Trim — ရွေးထားတဲ့ အပိုင်းတွေကိုပဲ ဖြတ်ယူပြီး တွဲမယ်
        _tr = []
        for p_ in (stl.get("trim") or [])[:40]:
            try:
                a_, b_ = float(p_[0]), float(p_[1])
                if b_ - a_ >= 0.4: _tr.append((max(0.0, a_), b_))
            except Exception: pass
        if _tr and src:
            parts = []
            for k_, (a_, b_) in enumerate(_tr):
                P(job, 1, "run", f"Video ဖြတ်နေ {k_ + 1}/{len(_tr)}", (k_ + .5) / len(_tr))
                pp = os.path.join(d, f"trim{k_}.mp4")
                sh(["ffmpeg", "-y", "-ss", f"{a_:.3f}", "-t", f"{b_ - a_:.3f}", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", pp])
                parts.append(pp)
            if len(parts) == 1: src = parts[0]
            else:
                lt = os.path.join(d, "trim.txt"); open(lt, "w").write("".join(f"file '{p}'\n" for p in parts))
                src = os.path.join(d, "trimmed.mp4")
                sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lt, "-c", "copy", src])
        dur, sw, sh_ = probe(src)
        if dur < 3: raise RuntimeError("Video တိုလွန်းပါတယ် (၃ စက္ကန့်အထက် လိုပါတယ်)")
        if dur > MAX_MIN * 60 + 5: raise RuntimeError(f"Video က {MAX_MIN} မိနစ်ထက် ပိုရှည်နေပါတယ် ({dur/60:.1f} မိနစ်) — ✂ Trim နဲ့ ဖြတ်ပါ")
        P(job, 1, "done", f"{dur/60:.1f} မိနစ်" if dur >= 60 else f"{dur:.0f} စက္ကန့်", 1)

        # 2) စကားကို စာထုတ် (AssemblyAI၊ မရရင် Gemini)
        P(job, 2, "run", "အသံထုတ်နေ", .05)
        aud = os.path.join(d, "audio.mp3")
        try: sh(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k", aud])
        except RuntimeError: raise RuntimeError("Video ထဲမှာ အသံ မတွေ့ပါ")
        sents, lc = [], ""
        try:
            sents, lc = aai_transcribe(aud, lang, lambda f, s: P(job, 2, "run", s, f), diar=False)
        except Exception as e:
            print("AssemblyAI မရ၊ Gemini နဲ့ စမ်းမယ်:", str(e)[:150])
        if not sents:
            sents = gemini_transcribe(aud, dur, lambda f, s: P(job, 2, "run", s, f))
            lc = lang if lang != "auto" else ""
        if not sents: raise RuntimeError("Video ထဲမှာ စကားပြောသံ မတွေ့ပါ")
        is_my = (lang == "my") or str(lc).lower().startswith(("my", "bur"))
        scenes = build_scenes(sents, dur)
        P(job, 2, "done", f"{len(sents)} ကြောင်း" + (f" ({lc})" if lc else ""), 1)

        # 3) အသံ ပြင်ဆင်
        P(job, 3, "run", "တန်းစီစောင့်နေ", 0)
        with GPU:
            P(job, 3, "run", "အသံ ပြင်ဆင်နေ", .2)
            if ai: ref = ensure_ai_voice(voice[3:])
            else:
                ref = os.path.join(d, "ref0.wav")
                sh(["ffmpeg", "-y", "-i", vsrc, "-vn", "-t", "30", "-ac", "1", "-ar", "44100", ref])
            P(job, 3, "run", "", .5)
            PR.clear(); RATE0[0] = None
            w = synth(CAL, ref, steps, spd, d)
        cps = max(6.0, min(20.0, len(CAL) / max(0.5, len(w) / SR))) * 0.8
        rate_cal = max(4.0, min(25.0, ncount(CAL) / max(0.5, speech_secs(w))))
        RATE0[0] = rate_cal / max(0.6, spd)
        P(job, 3, "done", f"{cps:.1f} လုံး/စက္ကန့်", 1)

        # 4) Gemini: မြန်မာစကားဆိုရင် စာလုံးပေါင်းပဲ ပြင်၊ တခြားဘာသာဆိုရင် မြန်မာလို ဘာသာပြန်
        P(job, 4, "run", "", 0)
        sp = [x for x in scenes if x["src"]]
        chunks = [sp[i:i + 15] for i in range(0, len(sp), 15)]
        def tr_chunk(chunk):
            if is_my:
                for x, t in zip(chunk, proofread([x["src"] for x in chunk])): x["t"] = t
            else:
                items = [{"text": x["src"], "chars": int(max(6, x["sd"] * cps))} for x in chunk]
                for x, t in zip(chunk, translate_batch(items, VC_HINT)): x["t"] = t
        with ThreadPoolExecutor(max_workers=1) as ex2:
            f2 = [ex2.submit(tr_chunk, c) for c in chunks]
            for k, fu in enumerate(f2):
                fu.result(); P(job, 4, "run", f"{k+1}/{len(chunks)}", (k + 1) / len(chunks))
        for x in scenes:
            if x["src"] and not x["t"]: x["t"] = fix_my(x["src"])
        N = len(scenes)
        P(job, 4, "done", f"{N} scene", 1)

        # 5) အသံထုတ် (GPU)  +  6) video ဖြတ်ညှိ (CPU)
        P(job, 5, "run", "တန်းစီစောင့်နေ", 0); P(job, 6, "run", f"0/{N}", 0)
        segs, full, tot, cnt, lk = [None] * N, [None] * N, [0.0] * N, {"n": 0}, threading.Lock()
        alens, auds = [0.0] * N, [None] * N
        texts = [x["t"] for x in scenes]
        def enc(i, ss, dd, s, need, au):
            n = max(FPS, math.ceil(dd * FPS)); dd = n / FPS
            sp_ = os.path.join(d, f"seg{i}.mp4")
            fc = (f"[0:v]setpts=(PTS-STARTPTS)/{s:.5f},fps={FPS},tpad=stop_mode=clone:stop_duration=30,"
                  f"scale='trunc(min(iw,1920)/2)*2':-2:flags=lanczos,setsar=1,format=yuv420p[v]")
            cmd = ["nice", "-n", "19", "ffmpeg", "-y", "-ss", f"{max(0.0, min(ss, dur - 0.5)):.3f}", "-t", f"{need:.3f}", "-i", src, "-an", "-filter_complex", fc, "-map", "[v]"]
            sh(cmd + ["-frames:v", str(n), "-c:v", "libx264", "-preset", "superfast", "-crf", "15", "-threads", "1", "-r", str(FPS), sp_])
            Tn = int(round(n * SR / FPS)); full[i] = np.pad(au, (0, max(0, Tn - len(au))))[:Tn]; segs[i] = sp_; tot[i] = dd
            with lk:
                cnt["n"] += 1; P(job, 6, "run", f"{cnt['n']}/{N}", cnt["n"] / N)
        ex = ThreadPoolExecutor(max_workers=2); futs = []
        with GPU:
            t_loop = time.time()
            for i, x in enumerate(scenes):
                for fu in futs:
                    if fu.done() and fu.exception(): raise fu.exception()
                eta = (time.time() - t_loop) / i * (N - i) if i >= 2 else 0
                P(job, 5, "run", f"{i}/{N}" + (f" · ~{int(eta // 60) + 1} မိနစ်ကျန်" if eta else ""), i / N)
                a, b, t = x["a"], x["b"], x["t"]
                if t:
                    T = b - a
                    au = synth(t, ref, steps, spd, d)
                    L = len(au) / SR
                    if L > T * 1.15 and T >= 1.0:
                        au = tempo(au, min(1.15, L / (T * 1.05)), d)      # နည်းနည်းပဲ မြန်ပေး
                else:
                    au = np.zeros(int(SR * (b - a)), np.float32)
                alens[i] = len(au) / SR
                auds[i] = au
        P(job, 5, "done", f"{N}/{N}", 1)
        plans = plan_all(scenes, [bool(texts[i]) for i in range(N)], alens, spd, dur)
        for i, p_ in enumerate(plans):
            if p_[0]: tot[i] = 0.0
            else: futs.append(ex.submit(enc, i, p_[1], p_[2], p_[3], p_[4], auds[i]))
        auds = None
        for fu in futs: fu.result()
        ex.shutdown(); P(job, 6, "done", f"{N}/{N}", 1)

        # 7) ပေါင်း → MP4 (ပုံက ပြန် encode မလုပ်တော့ဘူး)
        P(job, 7, "run", "ပေါင်းနေ", .05)
        lst = os.path.join(d, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in segs if p))
        vv, wv, out = os.path.join(d, "video.mp4"), os.path.join(d, "audio.wav"), os.path.join(d, "result.mp4")
        sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", vv])
        P(job, 7, "run", "အသံထည့်နေ", .5)
        sf.write(wv, np.concatenate([f_ for f_ in full if f_ is not None]), SR)
        total = sum(tot)
        sh(["ffmpeg", "-y", "-i", vv, "-i", wv, "-map", "0:v", "-c:v", "copy", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out])
        for p in [q for q in segs if q] + [vv, wv]:
            try: os.remove(p)
            except Exception: pass
        P(job, 7, "done", f"{mmss(total)} / မူရင်း {mmss(dur)}", 1); P(job, 8, "done", "", 1); job["out"] = out; job["done"] = True
    except Exception as e:
        traceback.print_exc(); job["err"] = str(e)[-500:]
        for x in job["steps"]:
            if x["s"] == "run": x["s"] = "err"
        refund_job(job)
    finally:
        job["fin"] = time.time(); job["upd"] = time.time()
        for p in (src, src0, vsrc):
            if not p: continue
            try: os.remove(p)
            except Exception: pass

# ---- web page
HTML = r'''<!DOCTYPE html><html lang="my"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voice Changer</title>
<style>
:root{color-scheme:light;--r:#e11d2e}*{box-sizing:border-box}
body{margin:0;background:#fff;color:#000;font-family:system-ui,"Noto Sans Myanmar","Padauk",sans-serif;line-height:1.7}
main{max-width:720px;margin:0 auto;padding:16px 14px 60px}h1{font-size:1.5rem;margin:6px 0 12px}
.card{border:1px solid #e6e6e6;border-radius:14px;padding:16px;margin-bottom:12px}h2{font-size:1.05rem;margin:0 0 10px}h3{font-size:.95rem;margin:14px 0 6px}
input[type=file],select,input[type=text],input[type=password]{width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;font:inherit;color:#000;background:#fff}
video,audio{width:100%;margin-top:10px}video{background:#000;border-radius:8px}
.btn{display:block;text-align:center;text-decoration:none;color:#fff;background:var(--r);border:0;border-radius:10px;padding:13px;width:100%;font:inherit;font-weight:700;cursor:pointer;margin-top:10px}
.btn.ol{background:#fff;color:#000;border:1px solid var(--r)}.btn:disabled{opacity:.4}.row{display:flex;gap:8px;align-items:center}
.note{font-size:.82rem;color:#000;margin-top:6px;opacity:.75}ul{list-style:none;padding:0;margin:0}
li{padding:8px;display:flex;gap:10px;align-items:center;border-bottom:1px dashed #e6e6e6;border-radius:8px;color:#777}.ic{width:24px;text-align:center}
.sp{display:inline-block;width:20px;height:20px;border:3px solid #e3e3e3;border-top-color:var(--r);border-radius:50%;animation:r .7s linear infinite;flex:none}
@keyframes r{to{transform:rotate(360deg)}}li.done{color:#000}li.done .ic{color:#1a9d5c}li.err{color:var(--r)}
li.run{background:#fff1f2;color:#000;font-weight:600}
.d{margin-left:auto;font-size:.82rem;text-align:right}
.live{display:flex;align-items:center;gap:14px;margin-bottom:8px}.big{width:46px;height:46px;border:5px solid #eee;border-top-color:var(--r);border-radius:50%;animation:r .8s linear infinite;flex:none}
.big.fin{animation:none;border-color:#1a9d5c;display:flex;align-items:center;justify-content:center;color:#1a9d5c;font-size:1.4rem}.big.bad{animation:none;border-color:var(--r);color:var(--r);display:flex;align-items:center;justify-content:center;font-size:1.4rem}
#pct{font-size:1.9rem;font-weight:800;line-height:1.1}#el{margin-left:auto;font-size:1rem;font-weight:700}
.bar{height:14px;background:#eee;border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;width:0;border-radius:10px;transition:width .6s;background:var(--r)}
.hide{display:none!important}pre{white-space:pre-wrap;font-size:.8rem;color:var(--r)}
.ctl{display:flex;align-items:center;gap:10px;margin-top:10px}.ctl input[type=range]{flex:1;accent-color:var(--r)}.pv{min-width:46px;text-align:right}
.top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.top .id{font-weight:800;letter-spacing:.5px;user-select:all}
.mini{border:1px solid var(--r);background:#fff;border-radius:8px;padding:2px 10px;font:inherit;cursor:pointer}
.hi{border:1px solid #e6e6e6;border-radius:12px;padding:12px;margin:8px 0}.hi b{word-break:break-all}.hi .bar{height:8px;margin-top:6px}
.drop{display:block;text-align:center;border:2px dashed #f1a1a8;background:#fff5f6;border-radius:16px;padding:24px 12px;cursor:pointer}
.drop .ico{width:48px;height:48px;margin:0 auto 8px;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08);display:flex;align-items:center;justify-content:center;font-size:1.4rem;color:var(--r)}
.or{display:flex;align-items:center;gap:10px;color:#666;font-size:.85rem;margin:12px 0}.or:before,.or:after{content:"";flex:1;height:1px;background:#e6e6e6}
.lrow{display:flex;gap:8px}.lbox{flex:1;display:flex;align-items:center;border:1px solid #ddd;border-radius:12px;padding:3px 3px 3px 10px;min-width:0}
.lbox input{border:0;padding:8px 4px;min-width:0;flex:1;outline:0;font:inherit;background:#fff;color:#000}
.pst{border:0;background:#fdecee;color:var(--r);font-weight:700;border-radius:9px;padding:8px 10px;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}
.dlb{border:0;background:#1b1b2b;color:#fff;font-weight:700;border-radius:12px;padding:0 14px;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}
.sw{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 0;border-bottom:1px solid #eee;font-weight:600}
.sw input{appearance:none;-webkit-appearance:none;width:46px;height:26px;border-radius:20px;background:#ccc;position:relative;flex:none;cursor:pointer;transition:.2s;margin:0}
.sw input:before{content:"";position:absolute;left:3px;top:3px;width:20px;height:20px;border-radius:50%;background:#fff;transition:.2s}
.sw input:checked{background:var(--r)}.sw input:checked:before{left:23px}
.pane{padding:6px 0 4px}
.trr{display:flex;align-items:center;gap:8px;padding:8px;border:1px solid #e6e6e6;border-radius:10px;margin-top:6px}.trr span{flex:1;font-weight:700;font-size:.9rem}.trr button{border:1px solid var(--r);background:#fff;border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit}
.vg{display:grid;grid-template-columns:1fr 1fr;gap:8px}.vc{border:1px solid #ddd;border-radius:12px;padding:10px;cursor:pointer;display:flex;align-items:center;gap:8px}.vc.on{border-color:var(--r);background:#fff1f2}.vc b{flex:1}.vc.wide{grid-column:1/-1}
.sec{display:flex;align-items:center;gap:8px;font-weight:800;margin:0 0 6px}.sec i{font-style:normal;width:26px;height:26px;border-radius:50%;background:var(--r);color:#fff;display:flex;align-items:center;justify-content:center;font-size:.85rem}
#gear{position:fixed;top:10px;right:12px;z-index:60;width:42px;height:42px;border-radius:50%;border:1px solid var(--r);background:#fff;font-size:1.25rem;cursor:pointer;box-shadow:0 1px 5px rgba(0,0,0,.25)}#gear.warn::after{content:'';position:absolute;top:3px;right:3px;width:11px;height:11px;border-radius:50%;background:var(--r)}
</style></head><body><button id="gear" title="Settings">⚙️</button><main>
<h1>🔊 Video အသံပြောင်း</h1>
<div class="card"><div class="top"><span>🆔 User ID:</span><span class="id" id="uid">…</span><button class="mini" id="cp">📋 Copy</button></div>
<div class="note" style="opacity:1">🎞 ထုတ်ခွင့်ကျန်: <span id="rem">…</span></div>
<div class="row"><button class="btn ol" id="hbtn">📜 History</button><button class="btn ol" id="newbtn">➕ Video အသစ်</button></div></div>

<div id="v-main">
<div class="card">
<div class="sec"><i>1</i> Video ထည့်ပါ</div>
<label class="drop" for="vf"><div class="ico">⬆</div><b id="dropt">Video ဖိုင် ရွေးပါ</b><div class="note">MP4, MOV, MKV · ၁၅ မိနစ်အထိ</div></label>
<input id="vf" type="file" accept="video/*" class="hide">
<div class="or">သို့မဟုတ် Link ထည့်ပါ</div>
<div class="lrow"><div class="lbox"><input id="lk" type="text" placeholder="TikTok / YouTube link ထည့်ပါ" autocomplete="off"><button class="pst" id="pst">📋 Paste</button></div><button class="dlb" id="dlb">🔗 ဒေါင်းမည်</button></div>
<div id="dlw" class="hide" style="margin-top:10px"><div class="bar"><i id="dpb"></i></div><div class="note" id="dtx"></div><pre id="der"></pre></div>
<details style="margin-top:8px"><summary class="note">🍪 cookies.txt (YouTube/Facebook က ပိတ်ရင်သာ ထည့်ပါ)</summary><input id="ck" type="file" accept=".txt,text/plain"></details>
<video id="pv" controls playsinline class="hide"></video>
<div class="note" id="vmsg"></div>
<div id="trw" class="hide" style="margin-top:10px"><div class="sw"><span>✂ လိုချင်တဲ့ အပိုင်းပဲ ဖြတ်မယ် (Trim)</span><input type="checkbox" id="t-tr"></div>
<div id="trp" class="pane hide">
<div class="note" style="opacity:1">အပေါ်က Video ကို ဖွင့်ကြည့်ပြီး လိုချင်တဲ့ အပိုင်းရဲ့ <b>စ</b> နေရာမှာ ရပ်ကာ <b>📍 စမှတ်</b>၊ <b>အဆုံး</b> နေရာမှာ ရပ်ကာ <b>📍 အဆုံးမှတ်</b> နှိပ်ပြီး <b>➕</b> နှိပ်ပါ။ အပိုင်းအများကြီး ထည့်လို့ရပါတယ်၊ ဘာမှမထည့်ရင် Video တစ်ခုလုံး ယူမယ်။</div>
<div class="row"><button class="btn ol" id="trm">◀ 0.5s</button><button class="btn ol" id="trp5">0.5s ▶</button></div>
<div class="row"><button class="btn ol" id="tra">📍 စမှတ် <b id="trav">00:00.0</b></button><button class="btn ol" id="trb">📍 အဆုံးမှတ် <b id="trbv">00:00.0</b></button></div>
<button class="btn" id="trn">➕ ဒီအပိုင်း ထည့်မယ်</button>
<div id="trl"></div><div class="note" id="trt" style="opacity:1;font-weight:700"></div></div></div>
<h3>🌐 Video ထဲက စကားပြောဘာသာ</h3><select id="lang"><option value="my" selected>မြန်မာ (Myanmar)</option><option value="auto">အလိုအလျောက် သိမယ် (Auto)</option><option value="en">English</option><option value="zh">Chinese</option><option value="ko">Korean</option><option value="ja">Japanese</option><option value="th">Thai</option><option value="hi">Hindi</option></select>
<div class="note">မြန်မာမဟုတ်ရင် မြန်မာလို ဘာသာပြန်ပြီး အသံထွက်ပေးမယ်။</div>
</div>

<div class="card">
<div class="sec"><i>2</i> အသံ ရွေးပါ</div>
<div class="vg" id="vg"></div>
<div class="note">▶ နှိပ်ရင် စမ်းနားထောင်လို့ရတယ်။ ပထမဆုံး အကြိမ်က အသံဖန်တီးလို့ ၁၀–၃၀ စက္ကန့် ကြာမယ်။</div>
<div id="ow" class="hide"><h3>🎤 သုံးချင်တဲ့ အသံဖိုင် (သန့်တဲ့ အသံ ၁၀–၃၀ စက္ကန့်)</h3><input id="af" type="file" accept="audio/*,video/*"><audio id="ap" controls class="hide"></audio>
<div class="note">ကိုယ့်အသံ (သို့) ခွင့်ပြုချက်ရတဲ့ အသံသာ သုံးပါ။</div></div>
<audio id="vta" controls class="hide"></audio>
</div>

<div class="card">
<div class="sec"><i>3</i> အမြန်နှုန်း</div>
<div class="ctl"><span>🐢</span><input type="range" id="sp" min="0.6" max="1.5" step="0.05" value="1"><span>🐇</span><b class="pv" id="spv">1.00x</b></div>
<div class="note">အသံကို ချိန်ရင် Video ကိုလည်း အသံနဲ့အညီ အလိုအလျောက် မြန်/နှေး ညှိပေးမယ် (1.00x = ပုံမှန်)။</div>
<button class="btn ol" id="vt">🔊 ရွေးထားတဲ့ အသံ၊ မြန်နှုန်းနဲ့ စမ်းနားထောင်</button>
<h3>⚡ အသံထုတ် အရည်အသွေး</h3><select id="q"><option value="4">အမြန်ဆုံး (နည်းနည်းလျော့)</option><option value="6" selected>ပုံမှန်</option><option value="10">အကောင်းဆုံး (နှေးတယ်)</option></select>
<div class="note">ℹ️ ၁၀ မိနစ် video ဆို ၁၅–၃၀ မိနစ်ခန့် ကြာနိုင်ပါတယ်။ Colab tab ကို မပိတ်ပါနဲ့။</div>
</div>
<button class="btn" id="go" disabled>▶ စတင်မည်</button>
<div class="note" id="gon">Video ထည့်ပြီး အသံရွေးပါ။</div>
</div>

<div id="v-job" class="card hide"><h2>လုပ်ဆောင်ချက် — <span id="jn"></span></h2>
<div class="live"><div class="big" id="bigsp"></div><div><div id="pct">0%</div><div class="note" id="cur" style="margin:0">စတင်နေပါတယ်…</div></div><div id="el">⏱ 00:00</div></div>
<div class="bar"><i id="pb"></i></div><div class="note" id="hb"></div>
<ul id="st" style="margin-top:8px"></ul><pre id="er"></pre>
<button class="btn ol" id="nx">➕ Video အသစ်ထုတ်မည်</button>
<div class="note">နောက်ကွယ်မှာ ဆက်လုပ်နေမယ်။ 📜 History မှာ ပြန်ကြည့်နိုင်တယ်။</div></div>

<div id="v-res" class="card hide"><h2>ရလဒ် — <span id="rn"></span></h2><video id="rv" controls playsinline></video>
<input id="fn" type="text" placeholder="ဖိုင်အမည် (မပေးလည်းရ)" style="margin-top:10px">
<a id="dl" class="btn">⬇ MP4 ဒေါင်းမည်</a>
<button class="btn ol" id="nx2">➕ Video အသစ်ထုတ်မည်</button></div>

<div id="v-set" class="card hide"><h2>⚙️ Settings — API Key</h2>
<h3>🔑 Gemini API key</h3><input id="kg" type="password" placeholder="AIza…" autocomplete="off"><div class="note" id="kgs" style="opacity:1;font-weight:700"></div>
<div class="note">ယူရန်: aistudio.google.com/apikey</div>
<h3>🔑 AssemblyAI API key</h3><input id="ka" type="password" placeholder="AssemblyAI key" autocomplete="off"><div class="note" id="kas" style="opacity:1;font-weight:700"></div>
<div class="note">ယူရန်: assemblyai.com/dashboard</div>
<button class="btn" id="ksave">💾 Save</button>
<div class="note">Save နှိပ်ရင် key မှန်မမှန် စစ်ပေးပါမယ်။ မှန်ရင် မှတ်ထားမှာမို့ နောက်ထပ် ထည့်စရာ မလိုတော့ပါဘူး။</div>
<button class="btn ol" id="kback">⬅ ပြန်သွားမည်</button></div>

<div id="v-hist" class="card hide"><h2>📜 History</h2><div id="hl"></div>
<button class="btn ol" id="hb8">➕ Video အသစ်ထုတ်မည်</button></div>
</main><script>
const $=i=>document.getElementById(i),VN=__VOICES__;
let vf=null,af=null,dlid=null,vid=null,upP=null,srcReady=false,dur=0,objU=null,watching=null,view='main',dT=null,curId=null,seen={},REM=null,sel='ai:'+Object.keys(VN)[0];
const ST={trim:[],trimOn:false};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const STEPS=["Video ရယူခြင်း","Video စစ်ဆေးခြင်း","AssemblyAI နဲ့ စကားကို စာထုတ်ခြင်း","အသံ ပြင်ဆင်ခြင်း","Gemini နဲ့ စာပြင်/ဘာသာပြန်ခြင်း","AI အသံထုတ်ခြင်း","Video ကို အသံနဲ့ ညှိခြင်း","MP4 ပေါင်းခြင်း","ပြီးစီးပါပြီ"];
const mm=s=>String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0');
const c=(v,a,b)=>Math.max(a,Math.min(b,v));
const rid=()=>Array.from(crypto.getRandomValues(new Uint8Array(12)),b=>b.toString(16).padStart(2,'0')).join('');
const MAXS=900;
function show(v){view=v;['main','job','res','hist','set'].forEach(n=>$('v-'+n).classList.toggle('hide',n!=v));if(v!='res')$('rv').pause();if(v!='main')$('pv').pause();window.scrollTo(0,0);if(v=='hist')loadHist()}

// ---- User ID + ထုတ်ခွင့်
async function refreshMe(){let m;try{m=await(await fetch('/api/me')).json()}catch(e){return}
$('uid').textContent=m.uid;REM=(m.limited&&m.ok)?m.rem:null;
$('rem').innerHTML=!m.ok?'<span style="color:#e11d2e">'+esc(m.msg||'Server ချိတ်မရပါ')+'</span>':m.limited?'<b>'+m.rem+'</b> ပုဒ်':'ကန့်သတ်မထား (စမ်းသပ်ခြင်း)'}
$('cp').onclick=()=>{try{navigator.clipboard.writeText($('uid').textContent);$('cp').textContent='✔ Copied'}catch(e){}};
$('hbtn').onclick=()=>show('hist');

async function upload(f,id,cb){const CH=8*1024*1024,n=Math.max(1,Math.ceil(f.size/CH));for(let i=0;i<n;i++){const fd=new FormData();fd.append('id',id);fd.append('idx',i);fd.append('n',n);fd.append('chunk',f.slice(i*CH,(i+1)*CH),'c');
const r=await fetch('/api/up',{method:'POST',body:fd});if(!r.ok)throw new Error('တင်လို့မရပါ ('+r.status+')');if(cb)cb((i+1)/n)}}
function effDur(){if(ST.trimOn&&ST.trim.length)return ST.trim.reduce((a,t)=>a+(t[1]-t[0]),0);return dur}
function upd(){const ed=effDur(),long=ed>MAXS;$('vmsg').style.color=long?'#e11d2e':'';
if(long)$('vmsg').textContent='⚠️ Video က ၁၅ မိနစ်ထက် ပိုရှည်နေပါတယ် ('+mm(ed)+') — ✂ Trim နဲ့ ဖြတ်ပါ';
const vok=sel!='own'||!!af;$('go').disabled=!(srcReady&&vok&&!long);
$('gon').textContent=!srcReady?'Video (သို့) Link ကို အရင်ထည့်ပါ':!vok?'သုံးချင်တဲ့ အသံဖိုင်ကို ထည့်ပါ':long?'Video ရှည်လွန်းနေပါတယ်':'အားလုံးအဆင်သင့်ပါ — စတင်လို့ရပါပြီ'}

// ---- Video / Link
function resetSrc(){if(objU){URL.revokeObjectURL(objU);objU=null}clearInterval(dT);dlid=null;vid=null;vf=null;srcReady=false;dur=0;$('vmsg').textContent='';$('dlw').classList.add('hide');$('der').textContent='';ST.trim=[];ST.trimOn=false;$('t-tr').checked=false;$('trp').classList.add('hide');$('trw').classList.add('hide');trList()}
function setSrc(u){const p=$('pv');p.src=u;p.classList.remove('hide');p.load()}
$('vf').onchange=e=>{const f=e.target.files[0]||null;if(!f)return;resetSrc();vf=f;vid=rid();srcReady=true;$('dropt').textContent='✔ '+f.name;objU=URL.createObjectURL(f);setSrc(objU);
upP=upload(f,vid,pr=>{if(!vf||vf!==f)return;$('vmsg').textContent=pr<1?'📤 Video ဖိုင်တင်နေ '+Math.round(pr*100)+'%':''});upP.catch(()=>{});upd()};
$('pst').onclick=async()=>{try{$('lk').value=await navigator.clipboard.readText()}catch(e){$('lk').focus();alert('Paste မရပါ — Link ကို ကြာကြာဖိပြီး Paste လုပ်ပါ')}};
$('dlb').onclick=async()=>{const u=$('lk').value;if(!/https?:\/\//i.test(u)){alert('Link မမှန်ပါ');return}
const ckf=$('ck').files[0];if(ckf){const cf=new FormData();cf.append('f',ckf);try{await fetch('/api/cookies',{method:'POST',body:cf})}catch(e){}}
const fd=new FormData();fd.append('url',u);let j={};try{j=await(await fetch('/api/dl',{method:'POST',body:fd})).json()}catch(e){}
if(!j.id){alert(j.msg||'Link မမှန်ပါ');return}
resetSrc();$('pv').pause();$('pv').classList.add('hide');$('dropt').textContent='Video ဖိုင် ရွေးပါ';$('vf').value='';
dlid=j.id;$('dlw').classList.remove('hide');$('dpb').style.width='0';$('dtx').textContent='ဒေါင်းနေပါတယ်… 0%';upd();pollDl()};
function pollDl(){clearInterval(dT);const id=dlid;dT=setInterval(async()=>{if(id!=dlid){clearInterval(dT);return}
let s;try{s=await(await fetch('/api/dlst/'+id)).json()}catch(e){return}
const p=Math.round((s.f||0)*100);$('dpb').style.width=p+'%';
$('dtx').textContent=s.state=='done'?'ဒေါင်းပြီးပါပြီ ✔':s.state=='err'?'ဒေါင်းမရပါ':'ဒေါင်းနေပါတယ်… '+p+'% '+(s.d||'');
if(s.state=='done'){clearInterval(dT);srcReady=true;setSrc('/api/dlv/'+id);upd()}
if(s.state=='err'){clearInterval(dT);$('der').textContent=s.msg||''}},1000)}
(function(){const p=$('pv');p.addEventListener('loadedmetadata',()=>{dur=p.duration||0;trA=0;trB=dur;$('trav').textContent=mmx(0);$('trbv').textContent=mmx(dur);$('trw').classList.remove('hide');trList();upd()});p.addEventListener('timeupdate',()=>{if(trStop!=null&&p.currentTime>=trStop){p.pause();trStop=null}})})();

// ---- ✂ Trim
let trStop=null,trA=0,trB=0;
const mmx=s=>{s=Math.max(0,s||0);const m=Math.floor(s/60);return String(m).padStart(2,'0')+':'+(s-m*60).toFixed(1).padStart(4,'0')};
function trList(){$('trl').innerHTML=ST.trim.map((t,i)=>'<div class="trr"><span>'+(i+1)+'. '+mmx(t[0])+' → '+mmx(t[1])+' ('+(t[1]-t[0]).toFixed(1)+'s)</span><button data-pv="'+i+'">▶</button><button data-dl="'+i+'">🗑</button></div>').join('');
$('trt').textContent=ST.trim.length?'ထုတ်မယ့် အလျား: '+mm(effDur())+' ('+ST.trim.length+' အပိုင်း)':'အပိုင်း မထည့်ရသေးပါ — Video တစ်ခုလုံး ယူမယ်'}
$('t-tr').onchange=e=>{ST.trimOn=e.target.checked;$('trp').classList.toggle('hide',!ST.trimOn);trList();upd()};
$('tra').onclick=()=>{trA=$('pv').currentTime;$('trav').textContent=mmx(trA)};
$('trb').onclick=()=>{trB=$('pv').currentTime;$('trbv').textContent=mmx(trB)};
$('trm').onclick=()=>{const p=$('pv');p.pause();p.currentTime=Math.max(0,p.currentTime-.5)};
$('trp5').onclick=()=>{const p=$('pv');p.pause();p.currentTime=Math.min(dur,p.currentTime+.5)};
$('trn').onclick=()=>{if(trB-trA<.5){alert('အဆုံးမှတ်က စမှတ်ထက် နောက်မှာ ရှိရမယ် (အနည်းဆုံး ၀.၅ စက္ကန့်)');return}
ST.trim.push([+trA.toFixed(2),+trB.toFixed(2)]);trA=trB;$('trav').textContent=mmx(trA);trList();upd()};
$('trl').onclick=e=>{const a=e.target.closest('[data-pv]'),d=e.target.closest('[data-dl]'),p=$('pv');
if(a){const t=ST.trim[+a.dataset.pv];p.currentTime=t[0];trStop=t[1];p.play()}
if(d){ST.trim.splice(+d.dataset.dl,1);trList();upd()}};

// ---- အသံ ရွေး / စမ်းနားထောင်
function drawV(){$('vg').innerHTML=Object.keys(VN).map(n=>'<div class="vc'+(sel=='ai:'+n?' on':'')+'" data-n="'+esc(n)+'"><b>'+esc(n)+' <span class="note">('+VN[n]+')</span></b><button class="mini" data-p="'+esc(n)+'">▶</button></div>').join('')+'<div class="vc wide'+(sel=='own'?' on':'')+'" data-n="__own"><b>🎤 ကိုယ်ကြိုက်တဲ့ အသံဖိုင် ထည့်မယ်</b></div>';
$('ow').classList.toggle('hide',sel!='own')}
drawV();
async function testVoice(voice,btn){const o=btn.textContent;btn.disabled=true;btn.textContent='⏳';
try{const f=new FormData();f.append('voice',voice);f.append('spd',$('sp').value);f.append('steps',$('q').value);
const r=await fetch('/api/vtest',{method:'POST',body:f});if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.msg||'မရပါ')}
const a=$('vta');a.src=URL.createObjectURL(await r.blob());a.classList.remove('hide');a.play().catch(()=>{})}catch(e){alert(String((e&&e.message)||e))}
btn.disabled=false;btn.textContent=o}
$('vg').onclick=e=>{const p=e.target.closest('[data-p]'),k=e.target.closest('[data-n]');
if(p){testVoice('ai:'+p.dataset.p,p);return}
if(k){sel=k.dataset.n=='__own'?'own':'ai:'+k.dataset.n;drawV();upd()}};
$('af').onchange=e=>{af=e.target.files[0]||null;if(af){$('ap').src=URL.createObjectURL(af);$('ap').classList.remove('hide')}upd()};
$('sp').oninput=()=>{$('spv').textContent=(+$('sp').value).toFixed(2)+'x'};
$('vt').onclick=async()=>{const b=$('vt');let v=sel;
if(sel=='own'){if(!af){alert('အသံဖိုင် အရင်ထည့်ပါ');return}b.disabled=true;b.textContent='⏳ အသံဖိုင် တင်နေ…';v=rid();try{await upload(af,v)}catch(e){alert(String(e.message||e));b.disabled=false;b.textContent='🔊 ရွေးထားတဲ့ အသံ၊ မြန်နှုန်းနဲ့ စမ်းနားထောင်';return}b.disabled=false}
await testVoice(v,b);b.textContent='🔊 ရွေးထားတဲ့ အသံ၊ မြန်နှုန်းနဲ့ စမ်းနားထောင်'};

// ---- စတင်မည်
$('go').onclick=async()=>{const V=vf,DL=dlid,L=$('lk').value.trim(),lang=$('lang').value,steps=$('q').value,spd=$('sp').value;
const VID=vid,UPP=upP,AF=af,own=sel=='own',STY=JSON.stringify({trim:(ST.trimOn?ST.trim:[])});
const isFile=!!V;
if(KS&&!(KS.g&&KS.a)){alert('⚙️ Settings မှာ Gemini နဲ့ AssemblyAI key ထည့်ပြီး Save နှိပ်ပါ');prevView='main';show('set');return}
$('go').disabled=true;let started=false,J=null;
try{const f0=new FormData();f0.append('name',isFile?V.name:L);const j0=await(await fetch('/api/new',{method:'POST',body:f0})).json();J=j0.id;if(!J)throw new Error(j0.msg||'စတင်မရပါ');
started=true;watching=J;show('job');poll();
let voice=sel;if(own){voice=rid();var a1=upload(AF,voice);a1.catch(()=>{})}
if(isFile)await UPP;if(own)await a1;
const fd=new FormData();fd.append('jid',J);fd.append('vid',isFile?VID:'');fd.append('dlid',isFile?'':DL);fd.append('voice',voice);
fd.append('lang',lang);fd.append('steps',steps);fd.append('style',STY);fd.append('spd',spd);
const j=await(await fetch('/api/start',{method:'POST',body:fd})).json();if(!j.ok)throw new Error(j.msg||'စတင်မရပါ');refreshMe();poll()
}catch(e){const m=String((e&&e.message)||e);if(started){const f=new FormData();f.append('msg',m);fetch('/api/fail/'+J,{method:'POST',body:f})}else alert(m)}
upd()};

// ---- progress
function paintJob(j){
const ico={wait:'○',run:'<i class="sp"></i>',done:'✔',err:'✖'},done=j.status=='done',bad=j.status=='err';
$('jn').textContent=j.name;
$('st').innerHTML=STEPS.map((n,i)=>{const x=j.steps[i]||{s:'wait'};const pc=x.s=='run'&&x.f?' '+Math.round(x.f*100)+'%':'';return `<li class="${x.s}"><span class="ic">${ico[x.s]}</span><span>${i+1}. ${n}</span><span class="d">${esc(x.d||'')}${pc}</span></li>`}).join('');
$('pb').style.width=j.pct+'%';$('pct').textContent=j.pct+'%';
const run=j.steps.map((x,i)=>x.s=='run'?STEPS[i]:null).filter(Boolean);
$('cur').textContent=done?'ပြီးစီးပါပြီ ✔':bad?'အမှားဖြစ်သွားပါတယ်':(run.join(' + ')||'စတင်နေပါတယ်…');
const bg=$('bigsp');bg.className='big'+(done?' fin':bad?' bad':'');bg.textContent=done?'✔':bad?'✖':'';
$('el').textContent='⏱ '+mm(j.el);$('er').textContent=j.err||'';
$('hb').textContent=(done||bad)?'':(j.ago>150?'🟡 ကြာနေပေမယ့် Server အလုပ်လုပ်နေဆဲပါ — နောက်ဆုံး update '+Math.round(j.ago)+' စက္ကန့်':'🟢 Server အလုပ်လုပ်နေပါတယ်');
document.title=(done?'✔ ':j.pct+'% ')+'Voice Changer'}
async function poll(){if(!watching)return;let j;try{j=await(await fetch('/api/job/'+watching)).json()}catch(e){return}
if(j.msg){watching=null;if(view=='job')show('main');return}
if(j.status!='run'&&!seen[j.id]){seen[j.id]=1;refreshMe()}
if(view!='job')return;paintJob(j);if(j.status=='done')openResult(j)}

// ---- ရလဒ်
function setLink(){if(!curId)return;const n=$('fn').value.trim().replace(/[\\\/:*?"<>|]/g,'');$('dl').href='/api/result/'+curId+'?dl=1'+(n?'&name='+encodeURIComponent(n):'')}
$('fn').oninput=setLink;
function openResult(j){curId=j.id;$('rn').textContent=j.name;$('rv').src='/api/result/'+j.id;$('fn').value='';$('fn').placeholder=String(j.name).replace(/\.[^.]+$/,'').slice(0,60);setLink();show('res')}

// ---- History
async function loadHist(){let l;try{l=await(await fetch('/api/jobs')).json()}catch(e){return}
if(view!='hist')return;window._hl=l;
$('hl').innerHTML=l.length?l.map(j=>{
const i=j.steps.findIndex(x=>x.s=='run');
const stt=j.status=='done'?'✅ ပြီးပါပြီ':j.status=='err'?'❌ မအောင်မြင်ပါ':'⏳ လုပ်နေဆဲ — '+j.pct+'%'+(i>=0?' · '+STEPS[i]:'');
const bar=j.status=='run'?'<div class="bar"><i style="width:'+j.pct+'%"></i></div>':'';
const err=j.status=='err'?'<pre>'+esc(j.err||'')+'</pre>':'';
const btn=j.status=='done'?'<div class="row"><button class="btn" data-v="'+j.id+'">▶ ကြည့်မည်</button><a class="btn ol" href="/api/result/'+j.id+'?dl=1">⬇ MP4 ဒေါင်း</a></div>':j.status=='run'?'<button class="btn ol" data-w="'+j.id+'">👁 အခြေအနေကြည့်မည်</button>':'';
return '<div class="hi"><b>'+esc(j.name)+'</b><div class="note">'+stt+' · ⏱ '+mm(j.el)+'</div>'+bar+err+btn+'</div>'}).join(''):'<div class="note">ထုတ်ထားတဲ့ video မရှိသေးပါ</div>'}
$('hl').onclick=e=>{const v=e.target.closest('[data-v]'),w=e.target.closest('[data-w]');
if(v){const id=v.dataset.v;const it=(window._hl||[]).find(x=>x.id==id);openResult({id:id,name:it?it.name:'video'})}
if(w){watching=w.dataset.w;show('job');poll()}};

function newVideo(){resetSrc();af=null;watching=null;
['vf','af','lk','ck'].forEach(i=>$(i).value='');
$('ap').removeAttribute('src');$('ap').classList.add('hide');$('vta').classList.add('hide');
const p=$('pv');p.pause();p.removeAttribute('src');p.load();p.classList.add('hide');$('rv').removeAttribute('src');
$('dropt').textContent='Video ဖိုင် ရွေးပါ';$('sp').value=1;$('spv').textContent='1.00x';
document.title='Voice Changer';upd();show('main')}
$('nx').onclick=$('nx2').onclick=$('hb8').onclick=$('newbtn').onclick=newVideo;

// ---- ⚙️ Settings (API key)
let KS=null,prevView='main';
function keyUI(){if(!KS)return;$('kgs').textContent=KS.g?'✅ ထည့်ပြီးသား ('+KS.gm+')':'⚠️ မထည့်ရသေးပါ';$('kas').textContent=KS.a?'✅ ထည့်ပြီးသား ('+KS.am+')':'⚠️ မထည့်ရသေးပါ';$('gear').classList.toggle('warn',!(KS.g&&KS.a))}
async function keyStatus(){try{KS=await(await fetch('/api/keys')).json()}catch(e){return}
let lg='',la='';try{lg=localStorage.getItem('k_g')||'';la=localStorage.getItem('k_a')||''}catch(e){}
if((!KS.g&&lg)||(!KS.a&&la)){const fd=new FormData();if(!KS.g&&lg)fd.append('gemini',lg);if(!KS.a&&la)fd.append('aai',la);try{const r=await(await fetch('/api/keys',{method:'POST',body:fd})).json();if(r.st)KS=r.st}catch(e){}}
keyUI()}
$('gear').onclick=()=>{if(view!='set')prevView=view;show('set');keyStatus()};
$('kback').onclick=()=>show(prevView||'main');
$('ksave').onclick=async()=>{const g=$('kg').value.trim(),a=$('ka').value.trim();if(!g&&!a){alert('Key အနည်းဆုံး တစ်ခု ထည့်ပါ');return}
const b=$('ksave');b.disabled=true;b.textContent='⏳ စစ်နေပါတယ်…';const fd=new FormData();if(g)fd.append('gemini',g);if(a)fd.append('aai',a);
try{const r=await(await fetch('/api/keys',{method:'POST',body:fd})).json();if(r.st)KS=r.st;
if(r.g){$('kgs').textContent=r.g.ok?'✅ မှန်ပါတယ် — သိမ်းပြီးပါပြီ ('+KS.gm+')'+(r.g.msg.includes('quota')?' · quota ပြည့်နေ':''):'❌ '+r.g.msg;if(r.g.ok){try{localStorage.setItem('k_g',g)}catch(e){}$('kg').value=''}}
if(r.a){$('kas').textContent=r.a.ok?'✅ မှန်ပါတယ် — သိမ်းပြီးပါပြီ ('+KS.am+')':'❌ '+r.a.msg;if(r.a.ok){try{localStorage.setItem('k_a',a)}catch(e){}$('ka').value=''}}
$('gear').classList.toggle('warn',!(KS.g&&KS.a))}
catch(e){alert('Save မရပါ — '+e.message)}
b.disabled=false;b.textContent='💾 Save'};
keyStatus().then(()=>{if(KS&&!(KS.g&&KS.a)&&view=='main'){prevView='main';show('set')}});

refreshMe();show('main');upd();
fetch('/api/jobs').then(r=>r.json()).then(l=>{if(l.some(j=>j.status=='run'))show('hist')}).catch(()=>{});
setInterval(()=>{poll();if(view=='hist')loadHist()},2000);
</script></body></html>'''

app = FastAPI()

@app.get("/")
def home(): return HTMLResponse(HTML.replace("__VOICES__", json.dumps({k: v[0] for k, v in AI_VOICES.items()}, ensure_ascii=False)))

@app.get("/api/me")
def me():
    r = srv(action="check", id=UID)
    ok = bool(r.get("ok"))
    return {"uid": UID, "ok": ok, "limited": bool(SERVER_URL), "rem": r.get("remaining") if ok else None, "msg": r.get("msg", "")}

@app.post("/api/new")
def new_job(name: str = Form("video")):
    r = srv(action="check", id=UID)
    if not r.get("ok"): return JSONResponse({"msg": r.get("msg", "Server ချိတ်မရပါ")}, status_code=503)
    if SERVER_URL and int(r.get("remaining") or 0) <= 0:
        return JSONResponse({"msg": f"video ထုတ်ခွင့် မကျန်တော့ပါ — Admin ကို သင့် User ID ({UID}) ပေးပြီး ထပ်ထည့်ခိုင်းပါ"}, status_code=402)
    jid = uuid.uuid4().hex[:16]
    steps = [{"s": "wait", "d": "", "f": 0} for _ in range(9)]
    steps[0] = {"s": "run", "d": "", "f": 0}
    JOBS[jid] = {"id": jid, "name": (name or "video")[:120], "created": time.time(), "fin": None, "steps": steps, "done": False, "err": None, "out": None, "upd": time.time()}
    return {"id": jid}

@app.post("/api/cookies")
def save_cookies(f: UploadFile):
    with open(COOKIES_FILE, "wb") as o: shutil.copyfileobj(f.file, o)
    return {"ok": True}

@app.post("/api/dl")
def dl_start(url: str = Form("")):
    m = re.search(r"https?://[^\s]+", url)
    if not m: return JSONResponse({"msg": "Link မမှန်ပါ"}, status_code=400)
    u = m.group(0).rstrip(".,)]}>\"'")
    did = uuid.uuid4().hex[:16]
    DL[did] = {"state": "run", "f": 0.0, "d": "", "msg": "", "path": None, "title": ""}
    threading.Thread(target=dl_worker, args=(did, u), daemon=True).start()
    return {"id": did}

@app.get("/api/dlst/{did}")
def dl_status(did: str):
    d = DL.get(did)
    if not d: return JSONResponse({"state": "err", "msg": "မတွေ့ပါ"}, status_code=404)
    return {"state": d["state"], "f": d["f"], "d": d["d"], "msg": d["msg"]}

@app.get("/api/dlv/{did}")
def dl_video(did: str):
    d = DL.get(did)
    if not d or not d.get("path") or not os.path.exists(d["path"]): return JSONResponse({"msg": "မတွေ့ပါ"}, status_code=404)
    return FileResponse(d["path"], media_type="video/mp4")

@app.post("/api/up")
def up(chunk: UploadFile, id: str = Form(""), idx: int = Form(0), n: int = Form(1)):
    if not re.fullmatch(r"[0-9a-f]{12,40}", id): return JSONResponse({"msg": "id မှားနေပါတယ်"}, status_code=400)
    with open(os.path.join(WORK, "up_" + id), "wb" if idx == 0 else "ab") as f: shutil.copyfileobj(chunk.file, f)
    return {"ok": True}


VT_TEXT = "မင်္ဂလာပါ။ ဒီအသံနဲ့ ပြောပြပါမယ်။ အသံက ဘယ်လိုနေလဲ နားထောင်ကြည့်ပါ။"
@app.post("/api/vtest")
def vtest(voice: str = Form(""), spd: float = Form(1.0), steps: int = Form(6)):
    ai = voice.startswith("ai:") and voice[3:] in AI_VOICES
    vs = os.path.join(WORK, "up_" + voice)
    if not ai and (not re.fullmatch(r"[0-9a-f]{12,40}", voice) or not os.path.exists(vs)):
        return JSONResponse({"msg": "အသံ မတွေ့ပါ၊ ပြန်ရွေး/ပြန်ထည့်ပါ"}, status_code=400)
    if not GPU.acquire(blocking=False): return JSONResponse({"msg": "Video ထုတ်နေတုန်းမို့ စမ်းနားထောင်လို့မရသေးပါ၊ အသံထုတ်တဲ့အဆင့်ပြီးမှ ပြန်စမ်းပါ"}, status_code=409)
    try:
        d = os.path.join(WORK, "vt_" + uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
        out = os.path.join(d, "t.wav")
        if ai: ref = ensure_ai_voice(voice[3:])
        else:
            ref = os.path.join(d, "ref.wav")
            sh(["ffmpeg", "-y", "-i", vs, "-vn", "-t", "30", "-ac", "1", "-ar", "44100", ref])
        PR.clear(); RATE0[0] = None
        sf.write(out, synth(VT_TEXT, ref, max(3, min(12, steps)), max(0.6, min(1.6, spd)), d), SR)
    except Exception as e:
        return JSONResponse({"msg": "အသံထုတ်မရပါ: " + str(e)[-200:]}, status_code=500)
    finally: GPU.release()
    return FileResponse(out, media_type="audio/wav")

@app.post("/api/start")
def start(jid: str = Form(""), vid: str = Form(""), dlid: str = Form(""), voice: str = Form(""),
          lang: str = Form("my"), steps: int = Form(6), style: str = Form("{}"), spd: float = Form(1.0)):
    if jid not in JOBS: return JSONResponse({"msg": "job မတွေ့ပါ"}, status_code=400)
    if not GEMINI_API_KEY or not ASSEMBLYAI_API_KEY: return JSONResponse({"msg": "API key မထည့်ရသေးပါ — ⚙️ Settings မှာ Gemini + AssemblyAI key ထည့်ပြီး Save နှိပ်ပါ"}, status_code=400)
    ai = voice.startswith("ai:") and voice[3:] in AI_VOICES
    chk = []
    if dlid:
        if dlid not in DL: return JSONResponse({"msg": "ဒေါင်းထားတဲ့ video မတွေ့ပါ"}, status_code=400)
        vid = ""
    else: chk.append(vid)
    if not ai: chk.append(voice)
    for x in chk:
        if not re.fullmatch(r"[0-9a-f]{12,40}", x) or not os.path.exists(os.path.join(WORK, "up_" + x)):
            return JSONResponse({"msg": "ဖိုင်မတွေ့ပါ၊ ပြန်တင်ပါ"}, status_code=400)
    if not re.fullmatch(r"[a-z]{2,3}(_[a-z]{2})?|auto", lang): lang = "my"
    u = srv(action="use", id=UID)      # ထုတ်ခွင့် ၁ ပုဒ် နှုတ်
    if not u.get("ok"): return JSONResponse({"msg": u.get("msg", "video ထုတ်ခွင့် မကျန်တော့ပါ")}, status_code=402)
    if u.get("ticket"): JOBS[jid]["ticket"] = u["ticket"]
    threading.Thread(target=run_job, args=(jid, vid, dlid, voice, lang, max(3, min(12, steps)), style, max(0.6, min(1.6, spd))), daemon=True).start()
    return {"ok": True}

@app.post("/api/fail/{jid}")
def fail(jid: str, msg: str = Form("")):
    j = JOBS.get(jid)
    if j and not j["done"] and not j["err"]:
        j["err"] = (msg or "မအောင်မြင်ပါ")[-500:]; j["fin"] = time.time(); j["upd"] = time.time()
        for x in j["steps"]:
            if x["s"] == "run": x["s"] = "err"
        refund_job(j)
    return {"ok": True}

@app.get("/api/job/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j: return {"msg": "မတွေ့ပါ"}
    now = time.time()
    st = "done" if j["done"] else "err" if j["err"] else "run"
    return {"id": j["id"], "name": j["name"], "status": st, "pct": pct(j), "steps": j["steps"], "err": j["err"], "ago": now - j["upd"], "el": (j["fin"] or now) - j["created"]}

@app.get("/api/jobs")
def jobs_list():
    now, out = time.time(), []
    for j in sorted(JOBS.values(), key=lambda x: -x["created"]):
        st = "done" if j["done"] else "err" if j["err"] else "run"
        out.append({"id": j["id"], "name": j["name"], "status": st, "pct": pct(j), "steps": j["steps"], "err": j["err"], "el": (j["fin"] or now) - j["created"]})
    return out

def clean_name(s): return re.sub(r'[\\/:*?"<>|\r\n]', "", s or "").strip()[:80]

@app.get("/api/result/{jid}")
def result(jid: str, dl: int = 0, name: str = ""):
    j = JOBS.get(jid)
    if not j or not j.get("out") or not os.path.exists(j["out"]): return JSONResponse({"msg": "မတွေ့ပါ"}, status_code=404)
    if dl:
        stem = clean_name(name) or clean_name(os.path.splitext(j["name"])[0]) or "recap"
        return FileResponse(j["out"], media_type="video/mp4", filename=stem + ".mp4")
    return FileResponse(j["out"], media_type="video/mp4")


# ---- API key (Website ⚙️ Settings)
def _mask(k): return (k[:4] + "…" + k[-4:]) if len(k) > 10 else ("•••" if k else "")

def _check_gemini(k):
    try:
        c = genai.Client(api_key=k)
        next(iter(c.models.list()), None)
        return True, "မှန်ပါတယ်"
    except Exception as e:
        t = str(e)
        if any(x in t for x in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "400", "403")): return False, "Key မှားနေပါတယ်"
        if "429" in t: return True, "မှန်ပါတယ် (quota ပြည့်နေ)"
        return False, "စစ်မရပါ: " + t[:100]

def _check_aai(k):
    try:
        r = requests.get(f"{AAI}/transcript", params={"limit": 1}, headers={"authorization": k}, timeout=20)
        if r.status_code == 200: return True, "မှန်ပါတယ်"
        if r.status_code in (401, 403): return False, "Key မှားနေပါတယ်"
        return False, f"စစ်မရပါ (HTTP {r.status_code})"
    except Exception as e: return False, "စစ်မရပါ: " + str(e)[:100]

def _keys_state():
    return {"g": bool(GEMINI_API_KEY), "a": bool(ASSEMBLYAI_API_KEY), "gm": _mask(GEMINI_API_KEY), "am": _mask(ASSEMBLYAI_API_KEY)}

@app.get("/api/keys")
def keys_status(): return _keys_state()

@app.post("/api/keys")
def keys_save(gemini: str = Form(""), aai: str = Form("")):
    global GEMINI_API_KEY, ASSEMBLYAI_API_KEY, gc, MODELS
    out, gk, ak, saved = {}, gemini.strip(), aai.strip(), False
    if gk:
        ok, m = _check_gemini(gk); out["g"] = {"ok": ok, "msg": m}
        if ok:
            GEMINI_API_KEY = gk; gc = genai.Client(api_key=gk); DEAD.clear(); COOL.clear(); TH.clear(); saved = True
            d = discover_models()
            if d: MODELS = d + [x for x in ("gemini-flash-latest",) if x not in d]
    if ak:
        ok, m = _check_aai(ak); out["a"] = {"ok": ok, "msg": m}
        if ok: ASSEMBLYAI_API_KEY = ak; saved = True
    if saved:
        try:
            json.dump({"gemini": GEMINI_API_KEY, "aai": ASSEMBLYAI_API_KEY}, open(KEYS_FILE, "w")); os.chmod(KEYS_FILE, 0o600)
        except Exception as e: print("key သိမ်းမရ:", str(e)[:100])
    out["st"] = _keys_state()
    return out


try: _SRV.should_exit = True; time.sleep(2)
except NameError: pass
_SRV = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
threading.Thread(target=_SRV.run, daemon=True).start()
time.sleep(3)
from google.colab import output
from IPython.display import display, HTML as IPY_HTML   # (HTML နာမည်က website စာမျက်နှာအတွက်ပါ — မထိပါနဲ့)

def start_tunnel():
    # website ကို cell ထဲမှာ ပြဖို့ public address ယူ (JS မလိုတဲ့နည်း — ဖုန်း view မှာလည်း အလုပ်လုပ်)
    try:
        exe = "/content/cloudflared"
        if not os.path.exists(exe):
            r = requests.get("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", timeout=180)
            r.raise_for_status(); open(exe, "wb").write(r.content); os.chmod(exe, 0o755)
        lg = open("/content/cf.log", "w")
        subprocess.Popen([exe, "tunnel", "--url", "http://localhost:8000", "--no-autoupdate", "--protocol", "http2"], stdout=lg, stderr=lg)
        t0 = time.time()
        while time.time() - t0 < 60:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/cf.log").read())
            if m:
                u = m.group(0)
                for _ in range(20):          # tunnel အသင့်ဖြစ်တဲ့အထိ စောင့်
                    try:
                        if requests.get(u + "/", timeout=10).status_code == 200: return u
                    except Exception: pass
                    time.sleep(2)
                return u
            time.sleep(1)
    except Exception as e:
        print("tunnel မရပါ:", str(e)[:120])
    return ""

site = start_tunnel()
ok = False
if site:
    try: ok = requests.get(site + "/", timeout=15).status_code == 200
    except Exception as e: print("tunnel စစ်မရ:", str(e)[:100])
purl = ""
try: purl = output.eval_js("google.colab.kernel.proxyPort(8000)") or ""      # Colab ကိုယ်ပိုင် link (အရန်)
except Exception: pass
_btn = '<a href="%s" target="_blank" rel="noopener" style="display:block;padding:14px;margin:8px 0;background:#e11d2e;color:#fff;border-radius:10px;font-size:18px;font-weight:700;text-align:center;text-decoration:none;font-family:sans-serif">%s</a>'
_h = ""
if site: _h += _btn % (site, "🌐 Website ဖွင့်မည် (Tab အသစ်) — ဒီခလုတ်ကို အရင်နှိပ်ပါ")
if purl: _h += _btn % (purl, "🔁 အရန် link (Colab)")
if _h: display(IPY_HTML(_h))
print("🌐 Website link:", site or "(မရပါ)")
print("🔁 Colab link  :", purl or "(မရပါ)")
if site and ok:
    print("✅ Website အဆင်သင့်ပါ — Colab tab ကို မပိတ်ပါနဲ့ (အောက်မှာ မပေါ်ရင် အပေါ်က ခလုတ်နဲ့ ဖွင့်ပါ)")
    display(IPY_HTML('<iframe src="%s" allow="clipboard-read; clipboard-write; fullscreen" style="width:100%%;height:1100px;border:0;background:#fff"></iframe>' % site))
else:
    print("⚠️ Tunnel မအဆင်သင့်သေးပါ — ၁–၂ မိနစ်စောင့်ပြီး အပေါ်က link ကို ဖွင့်ကြည့်ပါ၊ မရရင် cell ကို ပြန် Run ပါ")
    if not site:
        try: output.serve_kernel_port_as_iframe(8000, height="1100")
        except Exception as e: print("iframe မပေါ်ပါ:", str(e)[:100])

