# Local Studio - offline studio for weak GPU (Quadro M1200 4GB)
# Tabs: 1) Text-to-Image (SD 1.5 if installed) 2) Enhance 3) Upscale 4) Video
import os
import gc
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import gradio as gr

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
MODELS = BASE / "models"

# ---------- optional heavy deps (lazy) ----------
SD_AVAILABLE = False
SD_ERROR = ""
try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    SD_AVAILABLE = True
except Exception as e:
    SD_ERROR = str(e)[:500]

ESRGAN_AVAILABLE = False
try:
    import torch  # noqa: F401
    from realesrgan import RealESRGANer  # type: ignore
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    ESRGAN_AVAILABLE = True
except Exception:
    ESRGAN_AVAILABLE = False

DEFAULT_NEGATIVE = (
    "blurry, low quality, jpeg artifacts, distorted, deformed, "
    "extra fingers, bad hands, bad face, watermark, text, logo, "
    "oversaturated, overexposed, noisy, out of focus"
)

# ---------- كتالوج موديلات مجانية مفتوحة المصدر (تحميل مرة واحدة، ثم offline) ----------
MODEL_CATALOG = {
    "sd-turbo": {
        "label": "SD-Turbo (سريع: 1-4 خطوات — الأنسب لكارتك)",
        "repo": "stabilityai/sd-turbo",
        "github": "https://github.com/Stability-AI/generative-models",
        "license": "Stability AI Community (مجاني)",
        "size": "~4GB",
        "steps": 2, "cfg": 0.0,
    },
    "sd15": {
        "label": "SD 1.5 (جودة أعلى: 25-30 خطوة)",
        "repo": "runwayml/stable-diffusion-v1-5",
        "github": "https://github.com/CompVis/stable-diffusion",
        "license": "CreativeML Open RAIL-M (مجاني)",
        "size": "~4GB",
        "steps": 28, "cfg": 7.5,
    },
}

# أوزان GitHub المباشرة (upscale / وجوه) — روابط الإصدارات الرسمية
GITHUB_WEIGHTS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}

_pipes = {}


def get_sd_pipe(model_key="sd-turbo"):
    """Lazy load with 4GB optimizations. Downloads on first run (needs internet once), then offline."""
    if model_key in _pipes:
        return _pipes[model_key]
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    info = MODEL_CATALOG.get(model_key, MODEL_CATALOG["sd-turbo"])
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        info["repo"],
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if model_key == "sd15":
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cpu")
    except Exception:
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    _pipes[model_key] = pipe
    return pipe


def download_github_weight(name, progress=None):
    import urllib.request
    url = GITHUB_WEIGHTS[name]
    dest = MODELS / name
    MODELS.mkdir(exist_ok=True)
    if dest.exists():
        return f"موجود already: {name} ({dest.stat().st_size/1e6:.0f}MB)"
    def hook(b, bs, total):
        pass
    urllib.request.urlretrieve(url, str(dest))
    return f"تم التحميل: {name} ({dest.stat().st_size/1e6:.0f}MB) — يعمل offline الآن"


def models_status():
    lines = []
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        cached = {r.repo_id for r in cache.repos}
    except Exception:
        cached = set()
    for k, v in MODEL_CATALOG.items():
        hit = "✅ محمّل" if v["repo"] in cached or k in _pipes else "⬜ غير محمّل (يُحمّل أول مرة بإنترنت)"
        lines.append(f"- **{v['label']}** — {v['size']} — {v['license']} — {hit}\n  GitHub: {v['github']}")
    for name in GITHUB_WEIGHTS:
        p = MODELS / name
        hit = f"✅ موجود ({p.stat().st_size/1e6:.0f}MB)" if p.exists() else "⬜ غير موجود"
        lines.append(f"- **{name}** — {hit}\n  {GITHUB_WEIGHTS[name]}")
    return "\n".join(lines)


# ---------- فهم عربي + ستايلات (يعمل offline بعد أول تحميل) ----------
import re
_AR_RE = re.compile(r'[\u0600-\u06FF]')

def has_arabic(s: str) -> bool:
    return bool(s and _AR_RE.search(s))

_mt = None
def translate_ar(text: str) -> str:
    """ترجمة عربي->إنجليزي محليا (Helsinki-NLP, CPU)."""
    global _mt
    if not has_arabic(text):
        return text
    if _mt is None:
        from transformers import pipeline
        _mt = pipeline("translation_ar_to_en", model="Helsinki-NLP/opus-mt-ar-en", device=-1)
    out = []
    for m in re.finditer(r'[^\u0600-\u06FF]+|[\u0600-\u06FF]+', text):
        seg = m.group(0)
        if seg.strip() and _AR_RE.search(seg):
            seg = _mt(seg.strip())[0]["translation_text"]
        out.append(seg)
    return " ".join(s.strip() for s in out if s.strip()).strip()

STYLES = {
    "بدون ستايل": "",
    "سينمائي": ", cinematic lighting, film still, dramatic atmosphere, high detail",
    "واقعي فائق": ", ultra realistic, 8k, sharp focus, natural skin texture, professional photography",
    "أنمي": ", anime style, vibrant, detailed anime illustration, studio ghibli inspired",
    "كرتون 3D": ", 3d render, pixar style, soft lighting, octane render",
    "فنتازيا": ", epic fantasy art, intricate details, magical atmosphere, artstation",
}

def smart_prompt(prompt: str, style: str):
    """يفهم العربي (يترجمه) + يضيف بوستر الجودة حسب الستايل. يرجع (البرومبت النهائي، الترجمة)."""
    tr = translate_ar(prompt.strip()) if has_arabic(prompt) else prompt.strip()
    return (tr + STYLES.get(style, "")).strip(), (tr if tr != prompt.strip() else "")


# ---------- نص عربي سليم على التصاميم (PIL + Amiri — بدون تخبيص الموديلات) ----------
FONT_BOLD = BASE / "assets" / "fonts" / "Amiri-Bold.ttf"
FONT_REG = BASE / "assets" / "fonts" / "Amiri-Regular.ttf"

def draw_arabic(base: Image.Image, text: str, size: int = 64, color=(255, 255, 255),
                position: str = "أسفل", stroke: int = 2, bold: bool = True) -> Image.Image:
    import arabic_reshaper
    from bidi.algorithm import get_display
    from PIL import ImageDraw, ImageFont
    img = base.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    f = str(FONT_BOLD if bold else FONT_REG)
    try:
        font = ImageFont.truetype(f, int(size))
    except Exception:
        font = ImageFont.load_default()
    shaped = get_display(arabic_reshaper.reshape(text))
    bb = d.textbbox((0, 0), shaped, font=font, stroke_width=int(stroke))
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    W, H = img.size
    x = (W - tw) // 2
    y = {"أعلى": int(H * 0.08), "وسط": (H - th) // 2, "أسفل": int(H * 0.85) - th}.get(position, int(H * 0.85) - th)
    d.text((x, y), shaped, font=font, fill=tuple(color), stroke_width=int(stroke), stroke_fill=(0, 0, 0))
    return img


def make_design(bg: Image.Image, model_key, prompt, style, ar_text, size, seed,
                fsize, color_hex, position, stroke):
    """تصميم كامل: خلفية مولّدة أو مرفوعة + نص عربي واضح."""
    if bg is None:
        if not (prompt or "").strip():
            raise gr.Error("ارفع خلفية أو اكتب وصفا لتوليدها.")
        bg, _ = generate_image(model_key, prompt, "", 2 if model_key == "sd-turbo" else 28,
                               0.0 if model_key == "sd-turbo" else 7.5, size, int(seed),
                               style=style, auto_hq=False)
    else:
        bg = bg.convert("RGB")
    if (ar_text or "").strip():
        c = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
        bg = draw_arabic(bg, ar_text.strip(), int(fsize), c, position, int(stroke))
    p = OUT / f"design_{int(time.time())}.png"
    bg.save(p)
    return bg, str(p)


def generate_image(model_key, prompt, negative, steps, cfg, size, seed, style="بدون ستايل", auto_hq=False, quality="✅ قياسي"):
    if quality in QUALITY_PRESETS:
        q = QUALITY_PRESETS[quality]
        model_key, steps, cfg, auto_hq = q["model"], q["steps"], q["cfg"], q["hq"]
    if not SD_AVAILABLE:
        raise gr.Error(f"مكتبة التوليد غير مثبتة. ثبت requirements-full ثم أعد التشغيل. التفاصيل: {SD_ERROR}")
    if not prompt or not prompt.strip():
        raise gr.Error("اكتب وصف الصورة (prompt) أولا.")
    import torch
    w, h = {"512x512": (512, 512), "512x768": (512, 768), "768x512": (768, 512)}[size]
    # Quadro M1200 4GB: امنع أحجام كبيرة تسبب OOM وتشويه
    if model_key == "sd-turbo":
        steps, cfg = int(min(int(steps), 4)), 0.0  # Turbo مصمم لـ 1-4 خطوات بدون CFG
    final_prompt, shown_tr = smart_prompt(prompt, style)
    pipe = get_sd_pipe(model_key)
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    out = pipe(
        prompt=final_prompt,
        negative_prompt=(negative or DEFAULT_NEGATIVE),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        width=w, height=h,
        generator=g,
    )
    img = out.images[0]
    if auto_hq:
        # تحسين تلقائي + تكبير 2x بعد التوليد (جودة أعلى بدون تدخل)
        img = apply_quality(img, auto_hq if isinstance(auto_hq, str) else "full")
    p = OUT / f"sd_{int(time.time())}_{seed}.png"
    img.save(p)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    note = f"الترجمة: {shown_tr}" if shown_tr else "البرومبت إنجليزي مباشرة."
    return img, str(p), note


# ---------- Enhance (no artifacts) ----------
def enhance_image(img: Image.Image, denoise: float, clarity: float, color_boost: float):
    """تحسين بدون توشية: denoise خفيف + CLAHE على L فقط + unsharp خفيف."""
    if img is None:
        raise gr.Error("ارفع صورة أولا.")
    src = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 1) denoise خفيف يحافظ على التفاصيل (قيم عالية = بلور وتوشية)
    if denoise > 0:
        h = int(3 + denoise * 4)  # 3..11
        src = cv2.fastNlMeansDenoisingColored(src, None, h, h, 7, 21)

    # 2) تباين لطيف على قناة L فقط (يمنع تشبع الألوان)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5 + color_boost, tileGridSize=(8, 8))
    l = clahe.apply(l)
    src = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 3) حدة خفيفة Unsharp (amount صغير لمنع الهالات البيضاء)
    if clarity > 0:
        amount = 0.35 + clarity * 0.45  # max ~0.8
        blur = cv2.GaussianBlur(src, (0, 0), 2.0)
        src = cv2.addWeighted(src, 1 + amount, blur, -amount, 0)

    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    res = Image.fromarray(rgb)
    p = OUT / f"enhanced_{int(time.time())}.png"
    res.save(p)
    return res, str(p)


# ---------- Upscale ----------
_esrgan = None

def upscale_image(img: Image.Image, scale: int, use_esrgan: bool):
    if img is None:
        raise gr.Error("ارفع صورة أولا.")
    scale = int(scale)
    if scale not in (2, 4):
        scale = 2

    # مسار Real-ESRGAN فقط لو متثبت + الوزن موجود، غير كده Lanczos عالي الجودة
    if use_esrgan and ESRGAN_AVAILABLE:
        try:
            global _esrgan
            import torch
            if _esrgan is None:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                w = str(MODELS / "RealESRGAN_x4plus.pth")
                if not os.path.exists(w):
                    raise RuntimeError("وزن RealESRGAN غير موجود في models/ — سيتم استخدام Lanczos.")
                _esrgan = RealESRGANer(scale=4, model_path=w, model=model,
                                       tile=256, tile_pad=10, pre_pad=0,
                                       half=torch.cuda.is_available())
            arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            out, _ = _esrgan.enhance(arr, outscale=scale)
            res = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            p = OUT / f"up_esrgan{scale}x_{int(time.time())}.png"
            res.save(p)
            return res, str(p)
        except Exception as e:
            gr.Warning(f"Real-ESRGAN فشل ({str(e)[:200]}) — تم التحويل لـ Lanczos.")

    # Lanczos4 + tiling ضمني + تنعيم خفيف للحواف لمنع التعرجات
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    # حد أمان للذاكرة على 16GB RAM: امنع أبعاد عملاقة
    max_side = 4096
    if max(w, h) * scale > max_side:
        s = max_side / (max(w, h) * scale)
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        gr.Warning(f"تم تصغير مؤقت لتفادي نفاد الذاكرة (حد {max_side}px).")
    up = cv2.resize(arr, (arr.shape[1] * scale, arr.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)
    # إزالة نويز خفيفة بعد التكبير فقط
    up_bgr = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)
    up_bgr = cv2.bilateralFilter(up_bgr, 3, 15, 15)
    res = Image.fromarray(cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB))
    p = OUT / f"up_lanczos{scale}x_{int(time.time())}.png"
    res.save(p, quality=95)
    return res, str(p)


# ---------- مستويات جودة جاهزة (إعدادات قوية مضبوطة لكارتك) ----------
QUALITY_PRESETS = {
    "⚡ سريع":      {"model": "sd-turbo", "steps": 1,  "cfg": 0.0, "hq": False, "note": "تجربة سريعة"},
    "✅ قياسي":     {"model": "sd-turbo", "steps": 2,  "cfg": 0.0, "hq": "enhance", "note": "توازن سرعة/جودة"},
    "💎 عالي":      {"model": "sd-turbo", "steps": 4,  "cfg": 0.0, "hq": "full", "note": "تحسين + تكبير 2x"},
    "👑 فائق":      {"model": "sd15",     "steps": 28, "cfg": 7.5, "hq": "full", "note": "أفضل جودة (أبطأ)"},
}

ENHANCE_LEVELS = {
    "خفيف": (0.4, 0.25, 0.6),
    "متوسط": (0.7, 0.4, 1.0),
    "قوي": (1.1, 0.6, 1.6),
}


def apply_quality(img: Image.Image, hq):
    if hq == "enhance":
        img, _ = enhance_image(img, 0.7, 0.4, 1.0)
    elif hq == "full":
        img, _ = enhance_image(img, 0.7, 0.4, 1.0)
        img, _ = upscale_image(img, 2, False)
    return img


# ---------- تفريغ الخلفية + تصدير (JPG / PNG شفاف) ----------
def remove_bg(img: Image.Image) -> Image.Image:
    from rembg import remove
    return remove(img.convert("RGB"))


def export_image(img: Image.Image, fmt: str, jpg_q: int, transparent: bool, prefix="export"):
    if img is None:
        raise gr.Error("لا توجد صورة للتصدير.")
    if transparent:
        img = remove_bg(img)  # RGBA
        fmt = "PNG"
    if fmt == "JPG":
        img = img.convert("RGB")
        p = OUT / f"{prefix}_{int(time.time())}.jpg"
        img.save(p, quality=int(jpg_q))
    else:
        p = OUT / f"{prefix}_{int(time.time())}.png"
        img.save(p)
    return Image.open(p), str(p)


def produce_image(prompt, style, quality, shape, fmt, jpg_q, transparent, seed):
    """إنتاج بصيغة جاهزة: توليد بالجودة المختارة ثم تصدير JPG/PNG (مفرغ اختياري)."""
    size = {"مربع": "512x512", "طولي": "512x768", "عرضي": "768x512"}[shape]
    img, _, note = generate_image("sd-turbo", prompt, "", 2, 0.0, size, int(seed), style, False, quality)
    out_img, out_path = export_image(img, fmt, int(jpg_q), bool(transparent), "studio")
    return out_img, out_path, note


def produce_design(bg, prompt, style, quality, ar_text, fsize, color_hex, position, fmt, jpg_q, transparent, seed):
    if bg is None:
        if not (prompt or "").strip():
            raise gr.Error("ارفع خلفية أو اكتب وصفا لتوليدها.")
        bg, _, _ = generate_image("sd-turbo", prompt, "", 2, 0.0, "512x512", int(seed), style, False, quality)
    else:
        bg = bg.convert("RGB")
    if (ar_text or "").strip():
        c = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
        bg = draw_arabic(bg, ar_text.strip(), int(fsize), c, position, 2)
    if transparent:
        bg = remove_bg(bg)
        fmt = "PNG"
    return export_image(bg, fmt, int(jpg_q), False, "design")


def produce_enhance(img, level, up2x, fmt, jpg_q, transparent):
    if img is None:
        raise gr.Error("ارفع صورة أولا.")
    d, c, k = ENHANCE_LEVELS[level]
    img, _ = enhance_image(img.convert("RGB"), d, c, k)
    if up2x:
        img, _ = upscale_image(img, 2, False)
    return export_image(img, fmt, int(jpg_q), bool(transparent), "enhanced")


# ---------- Video: cinematic from image + enhance video ----------
def image_to_video(img: Image.Image, seconds: int, fps: int, zoom: float, width: int, ar_text: str = ""):
    """فيديو سينمائي (Ken Burns زوم بطيء) من صورة + نص عربي اختياري ثابت على الفيديو."""
    if img is None:
        raise gr.Error("ارفع صورة أولا.")
    seconds = int(np.clip(int(seconds), 2, 15))
    fps = int(fps) if int(fps) in (24, 30) else 24
    base = img.convert("RGB")
    # ثبت العرض، واحسب الارت بنسبة 16:9 مع crop مركزي (يمنع التشوه)
    tw = int(width)
    th = int(tw * 9 / 16)
    bw, bh = base.size
    s = max(tw / bw, th / bh) * 1.25  # هامش للزوم بدون حواف سوداء
    base = base.resize((int(bw * s), int(bh * s)), Image.LANCZOS)
    n = seconds * fps
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    vw = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th))
    cx, cy = base.size[0] // 2, base.size[1] // 2
    for i in range(n):
        t = i / max(n - 1, 1)
        z = 1.0 + float(zoom) * t  # زوم بطيء للأمام
        cw, ch = int(tw / z * (base.size[0] / tw)), int(th / z * (base.size[1] / th))
        # crop من المركز ثم resize للمقاس النهائي
        x0 = max(cx - cw // 2, 0); y0 = max(cy - ch // 2, 0)
        crop = base.crop((x0, y0, x0 + cw, y0 + ch)).resize((tw, th), Image.LANCZOS)
        if (ar_text or "").strip():
            crop = draw_arabic(crop, ar_text.strip(), max(28, tw // 16), (255, 255, 255), "أسفل", 2)
        vw.write(cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR))
    vw.release()
    final = str(OUT / f"cinematic_{int(time.time())}.mp4")
    # remux بـ ffmpeg لو موجود لجودة أعلى (crf 18)، وإلا استخدم الأصلي
    try:
        import shutil, subprocess
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-c:v", "libx264",
                            "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", final],
                           check=True, capture_output=True)
        else:
            final = tmp.name
    except Exception:
        final = tmp.name
    return final, final


def enhance_video(video_path: str, scale: int, denoise: bool):
    """تكبير + تنقية فيديو إطار بإطار (Lanczos + denoise خفيف). بطيء لكن offline وآمن."""
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("ارفع فيديو أولا.")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nw, nh = w * int(scale), h * int(scale)
    if max(nw, nh) > 1920:
        gr.Warning("تم تحديد الحد الأقصى 1920px للبعد الأكبر حفاظا على الذاكرة والسرعة.")
        k = 1920 / max(nw, nh)
        nw, nh = int(nw * k), int(nh * k)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    vw = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (nw, nh))
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if denoise:
            fr = cv2.fastNlMeansDenoisingColored(fr, None, 3, 3, 7, 21)
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        vw.write(fr)
    cap.release(); vw.release()
    final = str(OUT / f"video_up{scale}x_{int(time.time())}.mp4")
    try:
        import shutil, subprocess
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-c:v", "libx264",
                            "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", final],
                           check=True, capture_output=True)
        else:
            final = tmp.name
    except Exception:
        final = tmp.name
    return final, final


with gr.Blocks(title="Local Studio - استوديو محلي") as demo:
    gr.Markdown("# 🎬 Local Studio — يعمل محليا Offline\nجهازك: Quadro M1200 4GB — الإعدادات مضبوطة لتفادي التوشية والـ OOM.")
    if not SD_AVAILABLE:
        gr.Markdown(f"⚠️ **وضع خفيف:** مكتبات التوليد (torch/diffusers) غير مثبتة — تبويبات التحسين والـ Upscale والفيديو تعمل الآن. لتفعيل توليد الصور ثبت `requirements-full.txt`. ({SD_ERROR[:150]})")
    gr.Markdown("# 🎬 الاستوديو — إنشاء وتحسين بضغطة واحدة\nاكتب وصفا بالعربي أو الإنجليزية، اختر الجودة والصيغة، ودوس توليد.")
    with gr.Tabs():
        with gr.Tab("🎨 إنشاء صورة"):
            pr = gr.Textbox(label="اوصف الصورة (عربي أو إنجليزي)", lines=3, placeholder="مثال: أسد ذهبي في صحراء وقت الغروب...")
            with gr.Row():
                sy = gr.Dropdown(choices=list(STYLES.keys()), value="واقعي فائق", label="الستايل")
                q1 = gr.Radio(choices=list(QUALITY_PRESETS.keys()), value="✅ قياسي", label="مستوى الجودة")
            with gr.Row():
                sh1 = gr.Radio(choices=["مربع", "طولي", "عرضي"], value="مربع", label="المقاس")
                fm1 = gr.Radio(choices=["JPG", "PNG"], value="JPG", label="نوع الملف")
                jq1 = gr.Slider(60, 100, 92, step=1, label="جودة JPG")
            with gr.Row():
                tr1c = gr.Checkbox(value=False, label="خلفية مفرغة PNG شفاف (للأشخاص والمنتجات)")
                sd1 = gr.Number(value=42, label="Seed (غيّره لنتيجة مختلفة)", precision=0)
            b1 = gr.Button("✨ توليد", variant="primary")
            tr1 = gr.Textbox(label="الفهم", interactive=False)
            im1 = gr.Image(label="النتيجة")
            f1 = gr.File(label="تحميل الملف")
            b1.click(produce_image, [pr, sy, q1, sh1, fm1, jq1, tr1c, sd1], [im1, f1, tr1])
        with gr.Tab("✍️ تصميم بنص عربي"):
            dg_bg = gr.Image(type="pil", label="صورة خلفية (اختياري — سيبها فاضية للتوليد)")
            dg_pr = gr.Textbox(label="وصف الخلفية", lines=2, placeholder="سماء ليلية بالنجوم...")
            dg_tx = gr.Textbox(label="النص العربي", lines=1, placeholder="كوكب الزحالف")
            with gr.Row():
                dg_sy = gr.Dropdown(choices=list(STYLES.keys()), value="سينمائي", label="الستايل")
                dg_q = gr.Radio(choices=list(QUALITY_PRESETS.keys()), value="✅ قياسي", label="الجودة")
            with gr.Row():
                dg_fs = gr.Slider(24, 160, 72, step=2, label="حجم الخط")
                dg_cl = gr.ColorPicker(value="#FFFFFF", label="اللون")
                dg_ps = gr.Dropdown(["أعلى", "وسط", "أسفل"], value="أسفل", label="المكان")
            with gr.Row():
                dg_fm = gr.Radio(choices=["JPG", "PNG"], value="PNG", label="نوع الملف")
                dg_jq = gr.Slider(60, 100, 95, step=1, label="جودة JPG")
                dg_tr = gr.Checkbox(value=False, label="خلفية مفرغة PNG شفاف (للأشخاص والمنتجات)")
            bdg = gr.Button("✨ عمل التصميم", variant="primary")
            imdg = gr.Image(label="التصميم")
            fdg = gr.File(label="تحميل الملف")
            bdg.click(produce_design, [dg_bg, dg_pr, dg_sy, dg_q, dg_tx, dg_fs, dg_cl, dg_ps, dg_fm, dg_jq, dg_tr, gr.Number(value=7, visible=False, precision=0)], [imdg, fdg])
        with gr.Tab("✨ تحسين صورة"):
            i2 = gr.Image(type="pil", label="ارفع الصورة")
            with gr.Row():
                lv2 = gr.Radio(choices=["خفيف", "متوسط", "قوي"], value="متوسط", label="قوة التحسين")
                up2 = gr.Checkbox(value=True, label="تكبير 2x مع التحسين")
            with gr.Row():
                fm2 = gr.Radio(choices=["JPG", "PNG"], value="JPG", label="نوع الملف")
                jq2 = gr.Slider(60, 100, 92, step=1, label="جودة JPG")
                tr2c = gr.Checkbox(value=False, label="خلفية مفرغة PNG شفاف (للأشخاص والمنتجات)")
            b2 = gr.Button("✨ تحسين", variant="primary")
            im2 = gr.Image(label="بعد التحسين")
            f2 = gr.File(label="تحميل الملف")
            b2.click(produce_enhance, [i2, lv2, up2, fm2, jq2, tr2c], [im2, f2])
        with gr.Tab("🎥 فيديو"):
            i4 = gr.Image(type="pil", label="صورة البداية")
            tx4 = gr.Textbox(label="نص عربي على الفيديو (اختياري)", placeholder="كوكب الزحالف")
            with gr.Row():
                du = gr.Radio([3, 5, 10], value=5, label="المدة (ثواني)")
                wd = gr.Radio([640, 960], value=960, label="الجودة (العرض)")
            b4 = gr.Button("✨ إنشاء فيديو", variant="primary")
            v4 = gr.Video(label="الفيديو")
            f4 = gr.File(label="تحميل")
            b4.click(image_to_video, [i4, du, gr.Number(value=24, visible=False, precision=0), gr.Number(value=0.12, visible=False), wd, tx4], [v4, f4])
            gr.Markdown("---")
            v5 = gr.Video(label="فيديو للتحسين (تنقية + تكبير)")
            b5 = gr.Button("تحسين الفيديو")
            v6 = gr.Video(label="بعد التحسين")
            f6 = gr.File(label="تحميل")
            b5.click(enhance_video, [v5, gr.Number(value=2, visible=False, precision=0), gr.Checkbox(value=True, visible=False)], [v6, f6])
        with gr.Tab("📦 الموديلات"):
            gr.Markdown("موديلات مجانية مفتوحة المصدر — تحميل مرة واحدة بإنترنت، ثم تعمل **offline**.")
            st_md = gr.Markdown(value="اضغط تحديث لعرض الحالة.")
            b_ref = gr.Button("تحديث الحالة")
            b_ref.click(lambda: models_status(), outputs=[st_md])
            with gr.Row():
                wname = gr.Dropdown(choices=list(GITHUB_WEIGHTS.keys()), value="RealESRGAN_x4plus.pth",
                                    label="وزن من GitHub (Real-ESRGAN للتكبير / GFPGAN للوجوه)")
                b_dl = gr.Button("تحميل الوزن", variant="primary")
            dl_out = gr.Textbox(label="نتيجة التحميل")
            b_dl.click(download_github_weight, [wname], [dl_out])
            gr.Markdown("روابط المشاريع الأصلية:\n- SD-Turbo: https://github.com/Stability-AI/generative-models\n- SD 1.5: https://github.com/CompVis/stable-diffusion\n- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN\n- GFPGAN: https://github.com/TencentARC/GFPGAN\n- RIFE (سلاسة فيديو): https://github.com/hzwer/Practical-RIFE")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
