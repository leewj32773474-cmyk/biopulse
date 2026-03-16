import os
import requests
import json
from google import genai # 최신 라이브러리 방식

GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# 최신 Gemini 클라이언트 설정
client = genai.Client(api_key=GEMINI_API_KEY)

def get_ai_summary(title, description):
    prompt = f"As a biotech industry analyst, provide a 3-sentence professional insight in English. Focus on R&D impact. News: {title} - {description}"
    try:
        # 최신 모델 gemini-2.0-flash 사용 (속도가 훨씬 빠릅니다)
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"AI Error: {e}")
        return "Analysis pending for this breakthrough news."

# 1. 뉴스 가져오기
url = f"https://gnews.io/api/v4/search?q=biotech+pharma&lang=en&token={GNEWS_API_KEY}"
news_data = requests.get(url).json()

# 2. 분석 진행
if 'articles' in news_data:
    for article in news_data['articles'][:5]:
        article['ai_analysis'] = get_ai_summary(article.get('title', ''), article.get('description', ''))

# 3. 저장
with open('news.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=4)
