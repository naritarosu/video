# story_planner.py
import re
from typing import Dict, Tuple

import requests


# =========================
# Ollama client
# =========================
class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", model="qwen2.5:14b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages, temperature=0.6) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


# =========================
# Helpers: detect mixed Chinese / non-JP artifacts
# =========================
def contains_non_japanese_cjk(text: str) -> bool:
    suspicious = [
        "们", "为", "这", "那", "个", "同学", "饮料", "树荫", "凉快", "排队",
        "购买", "午后", "街头", "高中生", "饮", "队", "后", "们在"
    ]
    for s in suspicious:
        if s in text:
            return True
    if "，" in text:
        return True
    return False


# =========================
# Phase A: Story (JP) - Scene + Narration
# =========================
SYSTEM_STORY = """あなたは日本語の広告構成作家です。
30秒の動画PV向けに、4コマ構成（1=悩み/共感, 2=商品提示, 3=体験, 4=ベネフィット）を作ります。

【最重要】
- 出力は日本語のみ。
- 中国語（簡体字・繁体字）・英語は絶対に混ぜない。
- 「午后」「同学们」「饮料」など中国語表現は禁止。
- 性別や年齢を特定する語（少年/女性/男子高校生など）は使わず「主人公」で統一する

【出力フォーマット厳守】
以下の8行だけを出力してください（余計な説明は禁止）。各行は日本語1文で簡潔に。
各コマは「シーン（映像）」と「ナレーション（テロップ）」の2行セットです。

[1-Scene] ...
[1-Telop] ...
[2-Scene] ...
[2-Telop] ...
[3-Scene] ...
[3-Telop] ...
[4-Scene] ...
[4-Telop] ...

制約：
- [2-Telop] には必ず商品名を含める
- テーマに沿った具体的な状況にする
- 誇大表現（必ず/絶対/100%）は禁止
- 不自然な日本語・謎表現は禁止
"""

USER_STORY = """商品名とテーマから、4コマ構成を作ってください。

商品名：{product}
テーマ：{theme}

短く自然に。"""


def parse_story(raw: str) -> Dict[int, Dict[str, str]]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: Dict[int, Dict[str, str]] = {i: {"scene": "", "telop": ""} for i in [1, 2, 3, 4]}
    for ln in lines:
        m = re.match(r"\[(\d)\-(Scene|Telop)\]\s*(.+)", ln)
        if not m:
            continue
        idx = int(m.group(1))
        kind = m.group(2).lower()  # "scene" or "telop"
        text = m.group(3).strip()
        out[idx][kind] = text
    return out


def generate_story(llm_story: OllamaClient, product: str, theme: str) -> Dict[int, Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_STORY},
        {"role": "user", "content": USER_STORY.format(product=product, theme=theme)},
    ]

    last = ""
    for _ in range(5):
        raw = llm_story.chat(messages, temperature=0.35)
        last = raw
        data = parse_story(raw)

        ok = all(data[i]["scene"] and data[i]["telop"] for i in [1, 2, 3, 4])
        if not ok:
            messages.append({
                "role": "user",
                "content": "フォーマットが崩れています。必ず指定の8行のみで、[1-Scene]〜[4-Telop] を全て埋めてください。"
            })
            continue

        joined = "\n".join([
            data[1]["scene"], data[1]["telop"],
            data[2]["scene"], data[2]["telop"],
            data[3]["scene"], data[3]["telop"],
            data[4]["scene"], data[4]["telop"],
        ])

        if contains_non_japanese_cjk(joined):
            messages.append({
                "role": "user",
                "content": "中国語が混ざっています。必ず日本語のみで書き直してください（中国語・英語禁止）。フォーマット8行厳守。"
            })
            continue

        if product not in data[2]["telop"]:
            data[2]["telop"] = f"{product}で、その悩みをスッキリ解決。"

        return data

    raise RuntimeError(f"Failed to generate story.\nRaw:\n{last}")


# =========================
# Phase B: Image prompts (EN) for scenes 1,3,4
# =========================
SYSTEM_IMG = """You are a professional image-generation prompt engineer.

Your task is to propose image-generation prompts for three scenes:
1) customer pain/empathy
3) product experience
4) benefit/future state

CRITICAL RULES:
- There is exactly ONE protagonist in all images.
- NO other humans may appear anywhere in the image.
  (No crowds, no silhouettes, no background characters, no pedestrians.)
- Do NOT describe gender, age, ethnicity, hairstyle, clothing, or physical appearance.
  (These will be specified separately during image generation.)
- Focus ONLY on:
  background, environment, time of day, weather,
  the protagonist’s actions, posture, gestures, facial expression, and mood.
- Always refer to the character as "the protagonist".
- The protagonist must feel like the same person across all scenes.

STYLE & QUALITY:
- English only.
- Anime-style illustration.
- 16:9 composition.
- Clean, crisp line art, high-quality detailed background.
- No text, captions, logos, or readable brand names in the image.

Output only the required format. No explanations.
"""

USER_IMG = """Based on the following story, create image-generation prompts
for THREE scenes: Scene 1 (pain), Scene 3 (experience), Scene 4 (benefit).

Product: {product}
Theme: {theme}

Story reference:
[1-Scene] {s1_scene}
[1-Telop] {s1_telop}

[2-Scene] {s2_scene}
[2-Telop] {s2_telop}

[3-Scene] {s3_scene}
[3-Telop] {s3_telop}

[4-Scene] {s4_scene}
[4-Telop] {s4_telop}

PROMPT GUIDELINES (important):
- Describe the scene as a single illustration.
- Include: location, time of day, weather, lighting, camera framing.
- Describe the protagonist’s actions, posture, gestures, facial expression, and emotional state.
- Do NOT describe appearance attributes (gender, age, clothes, hair, etc.).
- The protagonist must be the ONLY human in the entire frame.
  ABSOLUTE: no other people anywhere (no crowds, no silhouettes, no background characters, no pedestrians).
- The product should appear only indirectly if needed (generic, non-branded).
- Always include: camera framing (e.g., medium shot / close-up), lens feel (cinematic), and depth of field.
- End each prompt with EXACTLY:
  "single protagonist only, only one person in the entire frame, no other people, no crowds, no silhouettes, 
  no background characters, anime-style, crisp lineart, high-detail background, cinematic lighting, depth of field, 
  16:9, no text, no logos".
OUTPUT FORMAT (strict):
[1-Prompt]
(1–3 sentences, English)

[3-Prompt]
(1–3 sentences, English)

[4-Prompt]
(1–3 sentences, English)
"""


def parse_prompts(raw: str) -> Dict[int, str]:
    out = {1: "", 3: "", 4: ""}

    def grab(idx: int) -> str:
        pattern = rf"\[{idx}\-Prompt\]\s*\n?(.*?)(?=\n\s*\[(?:1|3|4)\-Prompt\]|\Z)"
        m = re.search(pattern, raw, flags=re.DOTALL)
        if not m:
            return ""
        text = m.group(1).strip()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    out[1] = grab(1)
    out[3] = grab(3)
    out[4] = grab(4)
    return out


def generate_image_prompts(
    llm_img: OllamaClient,
    product: str,
    theme: str,
    story: Dict[int, Dict[str, str]],
    prefix: str,
) -> Dict[int, str]:
    messages = [
        {"role": "system", "content": SYSTEM_IMG},
        {"role": "user", "content": USER_IMG.format(
            product=product,
            theme=theme,
            s1_scene=story[1]["scene"], s1_telop=story[1]["telop"],
            s2_scene=story[2]["scene"], s2_telop=story[2]["telop"],
            s3_scene=story[3]["scene"], s3_telop=story[3]["telop"],
            s4_scene=story[4]["scene"], s4_telop=story[4]["telop"],
        )},
    ]

    last = ""
    for _ in range(4):
        raw = llm_img.chat(messages, temperature=0.7)
        last = raw
        data = parse_prompts(raw)
        ok = all(data[i] for i in [1, 3, 4])
        if ok:
            # normalize cola description if model leaks "clear"
            for k in [1, 3, 4]:
                data[k] = re.sub(r"\bclear fizzy liquid\b", "generic dark cola", data[k], flags=re.IGNORECASE)
                data[k] = re.sub(r"\bclear\b", "dark", data[k], flags=re.IGNORECASE)

            # ensure style suffix exists
            suffix = "anime-style, crisp lineart, high-detail background, cinematic lighting, depth of field, 16:9, no text, no logos."
            for k in [1, 3, 4]:
                if suffix.lower() not in data[k].lower():
                    data[k] = data[k].rstrip()
                    if not data[k].endswith((".", "!", "?")):
                        data[k] += "."
                    data[k] += " " + suffix

            # prepend appearance prefix (optional, user-defined)
            prefix_clean = (prefix or "").strip()
            if prefix_clean:
                if not prefix_clean.endswith((",", ";", ".")):
                    prefix_clean += ","
                for k in [1, 3, 4]:
                    data[k] = f"{prefix_clean} {data[k]}"

            return data

        messages.append({
            "role": "user",
            "content": "Format error. Output ONLY [1-Prompt], [3-Prompt], [4-Prompt] blocks. English only."
        })

    raise RuntimeError(f"Failed to generate image prompts.\nRaw:\n{last}")


def plan_story_and_prompts(product: str, theme: str, prefix: str) -> Tuple[Dict[int, Dict[str, str]], Dict[int, str]]:
    llm_story = OllamaClient(model="qwen2.5:14b")
    llm_img = OllamaClient(model="qwen2.5:14b")
    story = generate_story(llm_story, product, theme)
    prompts = generate_image_prompts(llm_img, product, theme, story, prefix or "")
    return story, prompts


# (CLIで単体確認したい場合だけ)
def _cli_main():
    print("=== Story Planner (JP story + EN prompts) ===\n")
    product = input("商品名: ").strip()
    theme = input("テーマ: ").strip()
    prefix = input("外見固定プロンプト（任意）: ").strip()
    story, prompts = plan_story_and_prompts(product, theme, prefix)

    print("\n--- ストーリー（JP）---")
    for i in [1, 2, 3, 4]:
        print(f"[{i}] シーン: {story[i]['scene']}")
        print(f"    テロップ: {story[i]['telop']}")

    print("\n--- Prompts（EN / 1,3,4）---")
    for i in [1, 3, 4]:
        print(f"[{i}] {prompts[i]}\n")


if __name__ == "__main__":
    # 通常は app.py から import して使います。単体実行もできるようにしています。
    _cli_main()
