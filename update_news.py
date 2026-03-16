import os
import requests
import json
import google.generativeai as genai

# 환경 변수에서 키 가져오기
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_summary(title, description):
    prompt = f"As a biotech researcher, summarize this news in 3 professional English sentences for a global audience. Focus on industry impact: Title: {title}, Description: {description}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Analysis in progress..."

# 1. 뉴스 가져오기
url = f"https://gnews.io/api/v4/search?q=biotech+pharma&lang=en&token={GNEWS_API_KEY}"
response = requests.get(url)
news_data = response.json()

# 2. 각 뉴스마다 AI 요약 추가
if 'articles' in news_data:
    for article in news_data['articles'][:5]: # 상위 5개 뉴스만 분석
        summary = get_ai_summary(article['title'], article['description'])
        article['ai_analysis'] = summary

# 3. 결과 저장
with open('news.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=4)
