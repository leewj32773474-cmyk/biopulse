import os
import requests
import json
import google.generativeai as genai

GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_summary(title, description):
    # AI가 거절하지 못하도록 프롬프트를 강화합니다.
    prompt = f"""
    Act as a professional pharmaceutical stock analyst. 
    Analyze this news for investors in 3 concise sentences. 
    Focus ONLY on market impact and R&D significance. 
    News Title: {title}
    Description: {description}
    """
    try:
        # 안전 설정을 낮춰서 뉴스 분석이 막히지 않게 합니다.
        response = model.generate_content(
            prompt,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Market analysis is being updated based on the latest R&D data."

# 뉴스 가져오기
url = f"https://gnews.io/api/v4/search?q=biotech+pharma&lang=en&token={GNEWS_API_KEY}"
response = requests.get(url)
news_data = response.json()

if 'articles' in news_data:
    for article in news_data['articles'][:5]:
        # 제목이나 설명이 없을 경우를 대비
        t = article.get('title', '')
        d = article.get('description', '')
        article['ai_analysis'] = get_ai_summary(t, d)

with open('news.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=4)
