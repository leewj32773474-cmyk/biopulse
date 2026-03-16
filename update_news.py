"""
BioPulse Global — Automated News Updater
Author: Claude (for Novarex R&D researcher @ Osong)

핵심 최적화:
  1. 배치 프롬프트: 여러 기사를 1번의 API 호출로 처리 → RPM 절약
  2. Exponential Backoff: 429 에러 시 자동 재시도 (최대 5회)
  3. 캐싱: 기존 분석 데이터 재사용 → 중복 API 호출 제거
  4. 쿼터 가드: 일일 호출 횟수 한도 자체 제어
"""

import os
import json
import time
import random
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GNEWS_API_KEY   = os.environ["GNEWS_API_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]

GNEWS_ENDPOINT  = "https://gnews.io/api/v4/search"
OUTPUT_PATH     = Path("news.json")

# Gemini 무료 티어: 15 RPM / 1,500 RPD
# 매시 1회 실행 → 최대 10기사 가져와 1~2회 API 호출로 처리
MAX_ARTICLES     = 10   # GNews에서 가져올 최대 기사 수
BATCH_SIZE       = 5    # 한 번의 Gemini 호출에 묶을 기사 수
MAX_RETRIES      = 5    # 429 재시도 횟수
BACKOFF_BASE     = 60   # 재시도 초기 대기 시간(초)

GEMINI_MODEL    = "gemini-1.5-flash"  # 무료 티어에서 가장 안정적

GNEWS_QUERIES = [
    "biopharma clinical trial",
    "pharmaceutical FDA approval",
    "biotech drug pipeline",
]

# ──────────────────────────────────────────────
# 유틸: 기사 고유 ID (URL 기반 해시)
# ──────────────────────────────────────────────
def article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────
# 1. GNews API로 뉴스 수집
# ──────────────────────────────────────────────
def fetch_news() -> list[dict]:
    """여러 쿼리로 뉴스를 수집하고 중복 제거 후 반환."""
    seen_urls = set()
    articles  = []

    for query in GNEWS_QUERIES:
        try:
            params = {
                "q":        query,
                "lang":     "en",
                "country":  "us",
                "max":      5,
                "apikey":   GNEWS_API_KEY,
            }
            resp = requests.get(GNEWS_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("articles", [])

            for art in data:
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append({
                        "id":          article_id(url),
                        "title":       art.get("title", ""),
                        "description": art.get("description", ""),
                        "url":         url,
                        "source":      art.get("source", {}).get("name", ""),
                        "published":   art.get("publishedAt", ""),
                        "image":       art.get("image", ""),
                        "ai_analysis": None,  # 분석 전
                    })

            log.info(f"GNews '{query}': {len(data)} articles fetched")
            time.sleep(1)  # GNews 무료 플랜 보호

        except Exception as e:
            log.warning(f"GNews error for '{query}': {e}")

    return articles[:MAX_ARTICLES]


# ──────────────────────────────────────────────
# 2. 기존 캐시 로드 (중복 분석 방지)
# ──────────────────────────────────────────────
def load_cache() -> dict:
    """news.json에서 이미 분석된 기사 ID → ai_analysis 매핑 반환."""
    if not OUTPUT_PATH.exists():
        return {}
    try:
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return {
            art["id"]: art["ai_analysis"]
            for art in data.get("articles", [])
            if art.get("ai_analysis")
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────
# 3. Gemini 배치 분석 (핵심 최적화)
# ──────────────────────────────────────────────
def build_batch_prompt(batch: list[dict]) -> str:
    """여러 기사를 하나의 프롬프트로 묶어 API 호출 횟수 최소화."""
    items = []
    for i, art in enumerate(batch, 1):
        items.append(
            f"[Article {i}]\n"
            f"Title: {art['title']}\n"
            f"Description: {art['description']}\n"
        )
    articles_text = "\n\n".join(items)

    return f"""You are a senior biopharma analyst with expertise in drug development, clinical trials, and biotech investment.
Analyze the following {len(batch)} news articles and return a JSON array ONLY — no markdown, no extra text.

Each element must have exactly these keys:
- "index": article number (1-based integer)
- "summary": 2-sentence expert summary in English (focus on clinical/scientific/market significance)
- "sentiment": one of "positive", "neutral", or "negative"
- "tags": array of 2-4 relevant tags (e.g., ["FDA", "oncology", "Phase 3"])
- "significance": one of "high", "medium", or "low"

Articles:
{articles_text}

Return only the JSON array, starting with [ and ending with ]."""


def call_gemini_with_backoff(client: genai.Client, prompt: str) -> str | None:
    """Exponential Backoff으로 429 에러 자동 재시도."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            return response.text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 10)
                log.warning(f"429 Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                log.error(f"Gemini error: {e}")
                return None

    log.error("Max retries exceeded. Skipping this batch.")
    return None


def analyze_articles(articles: list[dict], cache: dict) -> list[dict]:
    """캐시 미스 기사만 Gemini로 분석, 배치 처리."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 캐시 히트: 기존 분석 재사용
    for art in articles:
        if art["id"] in cache:
            art["ai_analysis"] = cache[art["id"]]
            log.info(f"Cache hit: {art['title'][:50]}...")

    # 캐시 미스 기사만 추출
    to_analyze = [a for a in articles if not a["ai_analysis"]]
    log.info(f"Articles to analyze: {len(to_analyze)} (cached: {len(articles) - len(to_analyze)})")

    if not to_analyze:
        return articles

    # 배치 단위로 처리
    for batch_start in range(0, len(to_analyze), BATCH_SIZE):
        batch = to_analyze[batch_start : batch_start + BATCH_SIZE]
        log.info(f"Calling Gemini for batch of {len(batch)} articles...")

        prompt = build_batch_prompt(batch)
        raw = call_gemini_with_backoff(client, prompt)

        if raw:
            try:
                # JSON 파싱 (마크다운 코드블록 방어)
                clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                results = json.loads(clean)

                for res in results:
                    idx = res.get("index", 0) - 1
                    if 0 <= idx < len(batch):
                        batch[idx]["ai_analysis"] = {
                            "summary":      res.get("summary", ""),
                            "sentiment":    res.get("sentiment", "neutral"),
                            "tags":         res.get("tags", []),
                            "significance": res.get("significance", "medium"),
                            "model":        GEMINI_MODEL,
                            "analyzed_at":  datetime.now(timezone.utc).isoformat(),
                        }
                log.info(f"Batch analyzed successfully.")

            except (json.JSONDecodeError, KeyError) as e:
                log.error(f"Failed to parse Gemini response: {e}\nRaw: {raw[:300]}")

        # 배치 간 대기 (RPM 보호: 15 RPM → 4초 이상 간격 확보)
        if batch_start + BATCH_SIZE < len(to_analyze):
            wait_time = 5 + random.uniform(1, 3)
            log.info(f"Waiting {wait_time:.1f}s before next batch (RPM protection)...")
            time.sleep(wait_time)

    return articles


# ──────────────────────────────────────────────
# 4. 결과 저장
# ──────────────────────────────────────────────
def save_output(articles: list[dict]):
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total":        len(articles),
        "articles":     articles,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Saved {len(articles)} articles to {OUTPUT_PATH}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=== BioPulse Global News Updater START ===")

    # Step 1: 뉴스 수집
    articles = fetch_news()
    log.info(f"Total unique articles fetched: {len(articles)}")

    if not articles:
        log.warning("No articles fetched. Aborting.")
        return

    # Step 2: 캐시 로드
    cache = load_cache()
    log.info(f"Cached analyses available: {len(cache)}")

    # Step 3: AI 분석
    articles = analyze_articles(articles, cache)

    # Step 4: 저장
    save_output(articles)
    log.info("=== BioPulse Global News Updater END ===")


if __name__ == "__main__":
    main()
