import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import io
import re
import textwrap
from datetime import datetime

# ─────────────────────────────────────────────
#  ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="添削くん",
    page_icon="✏️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  カスタムCSS（モバイルフレンドリー）
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kaisei+Decol:wght@400;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');
:root {
    --accent:#e84040; --bg:#fdf8f3; --card:#ffffff;
    --text:#1a1a2e; --muted:#888; --border:#e8e0d5;
}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg);font-family:'Noto Sans JP',sans-serif;}
.hero-header{text-align:center;padding:1.5rem 1rem 0.5rem;}
.hero-title{font-family:'Kaisei Decol',serif;font-size:clamp(2rem,8vw,3.2rem);color:var(--accent);
  letter-spacing:.05em;text-shadow:3px 3px 0 #fcd5d5;margin:0;}
.hero-sub{font-size:clamp(.75rem,3vw,.9rem);color:var(--muted);margin-top:.3rem;}
.mascot-area{display:flex;align-items:flex-end;justify-content:center;gap:1rem;margin:.5rem 0 1rem;}
.mascot-bubble{background:white;border:2px solid var(--accent);border-radius:16px 16px 16px 4px;
  padding:.6rem 1rem;font-size:.82rem;max-width:240px;color:var(--text);
  box-shadow:3px 3px 0 #fcd5d5;line-height:1.6;}
.mascot-emoji{font-size:3.5rem;filter:drop-shadow(2px 2px 0 #fcd5d5);}
.card{background:var(--card);border:1.5px solid var(--border);border-radius:16px;
  padding:1.5rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.mode-badge{display:inline-block;padding:.3rem .9rem;border-radius:999px;
  font-size:.78rem;font-weight:700;letter-spacing:.05em;margin-bottom:.8rem;}
.mode-math{background:#e3f2fd;color:#1565c0;border:1.5px solid #90caf9;}
.mode-essay{background:#f3e5f5;color:#6a1b9a;border:1.5px solid #ce93d8;}
.status-box{border-radius:12px;padding:.8rem 1.2rem;font-size:.85rem;margin:.5rem 0;line-height:1.6;}
.status-ok{background:#e8f5e9;border-left:4px solid #4caf50;color:#1b5e20;}
.status-info{background:#e3f2fd;border-left:4px solid #2196f3;color:#0d47a1;}
.status-warn{background:#fff3e0;border-left:4px solid #ff9800;color:#e65100;}
.status-wait{background:#fce4ec;border-left:4px solid #e91e63;color:#880e4f;}
.stButton>button{border-radius:12px!important;font-family:'Noto Sans JP',sans-serif!important;
  font-weight:700!important;font-size:1rem!important;padding:.6rem 1.2rem!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 4px 12px rgba(0,0,0,.15)!important;}
[data-testid="stSidebar"]{background:#fff8f8!important;}
.score-area{display:flex;gap:.8rem;flex-wrap:wrap;margin:.5rem 0;}
.score-chip{padding:.4rem 1rem;border-radius:999px;font-size:.85rem;font-weight:700;}
.chip-correct{background:#e8f5e9;color:#2e7d32;border:1.5px solid #81c784;}
.chip-wrong{background:#ffebee;color:#c62828;border:1.5px solid #ef9a9a;}
.chip-total{background:#f3e5f5;color:#6a1b9a;border:1.5px solid #ce93d8;}
@media(max-width:480px){.card{padding:1rem;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  定数・プロンプト
# ─────────────────────────────────────────────
MODEL_NAME = "gemini-1.5-flash"

SYSTEM_PROMPT_MATH = """
あなたは工業高校の数学教師AIです。
アップロードされた画像には生徒の解答が書かれています。

以下のJSON形式のみで返答してください（他のテキストは一切出力しないこと）：

{
  "summary": "全体的な講評（100字以内）",
  "total_questions": 問題の総数（整数）,
  "correct_count": 正解数（整数）,
  "annotations": [
    {
      "x_ratio": 0.0〜1.0（画像横幅に対する割合）,
      "y_ratio": 0.0〜1.0（画像縦幅に対する割合）,
      "mark": "○" または "×",
      "comment": "コメント（30字以内）。正解なら空文字でもよい"
    }
  ]
}

- ×の場合は必ずcommentで正解や解説を書いてください
- 問題が読み取れない場合はannotationsを空配列にしsummaryで説明してください
- JSONのみ返答。マークダウンの```json等は不要です
"""

SYSTEM_PROMPT_ESSAY = """
あなたは工業高校の国語・作文担当AIです。
アップロードされた画像には生徒の作文や記述解答が書かれています。

以下のJSON形式のみで返答してください（他のテキストは一切出力しないこと）：

{
  "summary": "全体的な講評と改善点（150字以内）",
  "total_questions": 設問の数（整数、なければ1）,
  "correct_count": 概ね正しく書けている設問数（整数）,
  "annotations": [
    {
      "x_ratio": 0.0〜1.0,
      "y_ratio": 0.0〜1.0,
      "mark": "○" または "△" または "×",
      "comment": "添削コメント（35字以内）"
    }
  ]
}

- ○は良い表現、△は改善推奨、×は誤りです
- JSONのみ返答。マークダウンの```json等は不要です
"""

MASCOT_MESSAGES = {
    "idle":    "プリントの写真をアップロードしてね！✨",
    "working": "AIが解析中です…そのまま待っていてね📝\nページを閉じたり更新しないこと！",
    "done":    "添削完了！画像をダウンロードしてTeamsに提出しよう📥",
    "error":   "うまく読み取れなかったよ。明るい場所でもう一度撮影してみて！",
}


# ─────────────────────────────────────────────
#  APIキー解決（Secrets優先 → サイドバー入力）
# ─────────────────────────────────────────────
def resolve_api_key(sidebar_input: str) -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return sidebar_input


# ─────────────────────────────────────────────
#  画像処理
# ─────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(result, -1, kernel)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


def call_gemini(api_key: str, pil_img: Image.Image, mode: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    system_prompt = SYSTEM_PROMPT_MATH if mode == "数学モード" else SYSTEM_PROMPT_ESSAY
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()
    response = model.generate_content(
        [system_prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
    )
    raw = response.text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def get_jp_font(size: int):
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_annotations(pil_img: Image.Image, annotations: list) -> Image.Image:
    img = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    mark_size = max(int(min(w, h) * 0.055), 28)
    font_mark = get_jp_font(mark_size)
    font_comment = get_jp_font(max(int(mark_size * 0.55), 14))
    COLORS = {
        "○": (30, 180, 60, 230),
        "△": (230, 140, 0, 230),
        "×": (220, 30, 30, 230),
    }
    for ann in annotations:
        try:
            cx = int(ann["x_ratio"] * w)
            cy = int(ann["y_ratio"] * h)
            mark = ann.get("mark", "○")
            comment = ann.get("comment", "").strip()
            color = COLORS.get(mark, (30, 180, 60, 230))
            r = mark_size // 2 + 4
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(255, 255, 255, 200), outline=color[:3] + (255,), width=3,
            )
            try:
                mb = font_mark.getbbox(mark)
                mw, mh = mb[2] - mb[0], mb[3] - mb[1]
                draw.text((cx - mw // 2, cy - mh // 2 - mb[1]), mark, font=font_mark, fill=color)
            except Exception:
                draw.text((cx, cy), mark, font=font_mark, fill=color)
            if comment:
                lines = textwrap.wrap(comment, width=16)
                pad = 6
                line_h = max(int(mark_size * 0.6), 16)
                box_w = max(len(l) for l in lines) * line_h // 2 + pad * 2
                box_h = len(lines) * line_h + pad * 2
                bx = min(cx + r + 4, w - box_w - 4)
                by = max(4, min(cy - box_h // 2, h - box_h - 4))
                draw.rounded_rectangle(
                    [bx, by, bx + box_w, by + box_h], radius=6,
                    fill=(255, 255, 200, 200), outline=color[:3] + (200,), width=2,
                )
                for i, line in enumerate(lines):
                    draw.text((bx + pad, by + pad + i * line_h), line,
                              font=font_comment, fill=(40, 40, 40, 255))
        except Exception:
            continue
    return Image.alpha_composite(img, overlay).convert("RGB")


def pil_to_download_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_filename(mode: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    prefix = "数学" if mode == "数学モード" else "作文"
    return f"添削済み_{prefix}_{date_str}.jpg"


# ─────────────────────────────────────────────
#  セッション初期化
# ─────────────────────────────────────────────
for key, default in [
    ("mascot_state", "idle"),
    ("result_img", None),
    ("result_json", None),
    ("processed_img", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
#  サイドバー
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 設定")
    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secrets_key = ""

    if secrets_key:
        st.markdown(
            '<div style="background:#e8f5e9;border-radius:8px;padding:.5rem .8rem;'
            'font-size:.8rem;color:#2e7d32;">✅ APIキーは管理者が設定済みです</div>',
            unsafe_allow_html=True,
        )
        sidebar_key_input = ""
    else:
        sidebar_key_input = st.text_input(
            "🔑 Gemini APIキー",
            type="password",
            placeholder="AIzaSy...",
            help="Google AI Studio から取得してください",
        )

    mode = st.radio("📚 採点モード", ["数学モード", "作文モード"], index=0)
    st.markdown("---")
    st.markdown("#### 📷 撮影のコツ")
    st.markdown("""
- 明るい場所で撮影する
- プリント全体が入るように
- できるだけ真上から撮る
- ブレに注意！
""")
    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa'>添削くん v1.1<br>Gemini 1.5 Flash 使用</small>",
        unsafe_allow_html=True,
    )

api_key = resolve_api_key(sidebar_key_input)


# ─────────────────────────────────────────────
#  メイン画面
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">✏️ 添削くん</p>
  <p class="hero-sub">AIがプリントを自動採点・添削するWebアプリ</p>
</div>
""", unsafe_allow_html=True)

mascot_msg = MASCOT_MESSAGES.get(st.session_state.mascot_state, MASCOT_MESSAGES["idle"])
st.markdown(f"""
<div class="mascot-area">
  <span class="mascot-emoji">🤖</span>
  <div class="mascot-bubble">{mascot_msg}</div>
</div>
""", unsafe_allow_html=True)

badge_class = "mode-math" if mode == "数学モード" else "mode-essay"
badge_icon = "📐" if mode == "数学モード" else "✍️"
st.markdown(
    f'<span class="mode-badge {badge_class}">{badge_icon} {mode}</span>',
    unsafe_allow_html=True,
)

# アップロード
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📸 プリントをアップロード")
uploaded_file = st.file_uploader(
    "JPG / PNG など",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    pil_raw = Image.open(uploaded_file)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🖼️ アップロード画像（補正後プレビュー）")
    processed = preprocess_image(pil_raw)
    st.session_state.processed_img = processed
    st.image(processed, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not api_key:
        st.markdown(
            '<div class="status-box status-warn">⚠️ サイドバーに Gemini APIキーを入力してください</div>',
            unsafe_allow_html=True,
        )
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start = st.button("🚀 AIに添削してもらう！", use_container_width=True, type="primary")

        if start:
            st.session_state.mascot_state = "working"
            st.session_state.result_img = None
            st.session_state.result_json = None

            st.markdown("""
            <div class="status-box status-wait">
            ⏳ <b>AIが解析中です。このまま待っていてください。</b><br>
            ・処理には10〜30秒かかることがあります<br>
            ・ページを閉じたり、更新（リロード）したりしないこと！<br>
            ・混雑時は少し長くかかる場合があります
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Gemini が解析中…少々お待ちください"):
                try:
                    result = call_gemini(api_key, processed, mode)
                    annotated = draw_annotations(processed, result.get("annotations", []))
                    st.session_state.result_img = annotated
                    st.session_state.result_json = result
                    st.session_state.mascot_state = "done"
                except json.JSONDecodeError as e:
                    st.session_state.mascot_state = "error"
                    st.error(f"AIの返答をJSONとして解析できませんでした: {e}")
                except Exception as e:
                    st.session_state.mascot_state = "error"
                    err = str(e)
                    if "API_KEY" in err.upper() or "invalid" in err.lower():
                        st.error("APIキーが正しくありません。先生に確認してください。")
                    elif "quota" in err.lower():
                        st.error("現在アクセスが集中しています。5分ほど待ってから再試行してください。")
                    else:
                        st.error(f"エラーが発生しました: {err}")
            st.rerun()

# 結果表示
if st.session_state.result_img is not None and st.session_state.result_json is not None:
    res = st.session_state.result_json
    annotated_img = st.session_state.result_img

    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📋 採点結果")
    total = res.get("total_questions", 0)
    correct = res.get("correct_count", 0)
    wrong = max(0, total - correct)
    score_pct = int(correct / total * 100) if total > 0 else 0
    st.markdown(f"""
    <div class="score-area">
      <span class="score-chip chip-correct">✅ 正解: {correct}問</span>
      <span class="score-chip chip-wrong">❌ 不正解: {wrong}問</span>
      <span class="score-chip chip-total">📊 得点率: {score_pct}%</span>
    </div>
    """, unsafe_allow_html=True)
    summary = res.get("summary", "")
    if summary:
        st.markdown(
            f'<div class="status-box status-info">💬 <b>AIからのコメント：</b><br>{summary}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ✏️ 添削済み画像")
    st.image(annotated_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    dl_bytes = pil_to_download_bytes(annotated_img)
    filename = make_filename(mode)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 添削画像をダウンロード",
            data=dl_bytes,
            file_name=filename,
            mime="image/jpeg",
            use_container_width=True,
        )

    st.markdown(
        f'<div class="status-box status-ok">'
        f'✅ <b>{filename}</b> をダウンロードして、Microsoft Teams の課題に添付して提出しよう！'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔍 AIの返答（JSON）を確認する（先生向け）"):
        st.json(res)
