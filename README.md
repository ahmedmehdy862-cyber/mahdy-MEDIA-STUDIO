# 🎬 Local Studio — استوديو محلي عربي يعمل Offline

استوديو توليد وتحسين صور وفيديو **يعمل على جهازك بدون إنترنت** (بعد تحميل الموديلات أول مرة)،
مصمم للأجهزة الضعيفة (مجرّب على NVIDIA Quadro M1200 4GB)، ويفهم **العربية والإنجليزية**،
ويضيف **نصوصا عربية سليمة** على التصاميم والفيديوهات.

> English: offline Arabic-first AI studio — text-to-image (SD-Turbo / SD 1.5),
> Arabic typography overlay, enhance, upscale, cinematic video. Tested on 4GB VRAM.

## ✨ المميزات

| الميزة | الوصف |
|---|---|
| 🖼️ توليد صور | SD-Turbo (خطوتان — سريع) و SD 1.5 (جودة أعلى) + تحسين وتكبير تلقائي |
| 🌍 فهم عربي | البرومبت العربي يُترجم محليا للإنجليزية (Helsinki-NLP، يعمل offline) |
| ✍️ نص عربي سليم | خط أميري + تشكيل صحيح (موديلات التوليد لا ترسم العربية، نضيفها بدقة) |
| ✨ تحسين بدون توشية | denoise خفيف + CLAHE على الإضاءة فقط + حدة محسوبة |
| 🔍 تكبير Upscale | Lanczos 2x/4x + Real-ESRGAN اختياري |
| 🎥 فيديو | سينمائي من صورة + نص عربي ثابت + تنقية وتكبير فيديو |
| 📦 موديلات مجانية | تبويب إدارة وتحميل من المصادر المفتوحة |

## 🖥️ المتطلبات

- Windows 10/11 + Python 3.10/3.11/3.12
- كارت NVIDIA 4GB+ (مجرّب على Quadro M1200) — أو CPU (أبطأ)
- مساحة ~10GB للموديلات + إنترنت **مرة واحدة** للتحميل

## 🚀 التشغيل

```bat
run.bat
```

ثم افتح: `http://127.0.0.1:7860` — لا تغلق النافذة السوداء أثناء الاستخدام.

تثبيت يدوي:

```bat
pip install -r requirements-min.txt   :: الواجهة والتحسين والفيديو
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors huggingface_hub sentencepiece arabic_reshaper python-bidi
```

## 📦 الموديلات (مجانية مفتوحة المصدر)

| الموديل | المصدر | الرخصة |
|---|---|---|
| SD-Turbo | https://github.com/Stability-AI/generative-models | Stability AI Community |
| SD 1.5 | https://github.com/CompVis/stable-diffusion | CreativeML Open RAIL-M |
| Real-ESRGAN | https://github.com/xinntao/Real-ESRGAN | BSD-3 |
| GFPGAN | https://github.com/TencentARC/GFPGAN | Apache-2.0 |
| RIFE | https://github.com/hzwer/Practical-RIFE | MIT |
| opus-mt-ar-en | https://github.com/Helsinki-NLP/Opus-MT | CC-BY-4.0 |
| Amiri font | https://github.com/google/fonts | OFL |

الكود في هذا المستودع تحت رخصة MIT. الموديلات والأوزان تخضع لرخص أصحابها (انظر الجدول).

## ⚠️ ملاحظات الأجهزة الضعيفة

- أول توليدة بطيئة (تحميل الموديل)، بعدها التوليد ثوانٍ — اترك التطبيق شغالا.
- لا تتجاوز 512px على كروت 4GB (إعدادات مضبوطة ضد OOM والتشويه).
- توليد الفيديو بالـ diffusion يحتاج 12GB+ — لذلك الفيديو هنا سينمائي + تحسين.
