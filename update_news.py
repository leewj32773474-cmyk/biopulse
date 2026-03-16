import os
import requests
import json
import time # 휴식 시간을 위한 라이브러리 추가
from google import genai

GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)

def get_ai_summary(title, description):
    # 무료 버전은 1분에 15번 정도만 가능하므로, 딜레이를 줍니다.
    time.sleep(5) 
    
    prompt = f"As a biotech industry analyst, provide a 3-sentence professional insight in English. Focus on R&D impact. News: {title} - {description}"
    try:
        # 모델을 가장 안정적인 gemini-1.5-flash로 살짝 변경합니다. (2.0은 가끔 쿼터가 꼬입니다)
        response = client.models.generate_content(
            model='gemini-1.5-flash', 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"AI Error: {e}")
        return "Market analysis is being updated based on the latest industry trends."

# 1. 뉴스 가져오기
url = f"https://gnews.io/api/v4/search?q=biotech+pharma&lang=en&token={GNEWS_API_KEY}"
try:
    news_data = requests.get(url).json()
except:
    news_data = {"articles": []}

# 2. 분석 진행
if 'articles' in news_data:
    # 한꺼번에 너무 많이 하면 또 에러가 나니, 상위 3개만 먼저 확실히 분석해 봅시다.
    for article in news_data['articles'][:3]:
        print(f"Analyzing: {article.get('title')[:30]}...")
        article['ai_analysis'] = get_ai_summary(article.get('title', ''), article.get('description', ''))

# 3. 저장
with open('news.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=4)
