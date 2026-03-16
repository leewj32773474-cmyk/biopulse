import os
import requests
import json
import google.generativeai as genai

GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_summary(title, description):
    try:
        # AI에게 연구원 수준의 분석을 요구합니다.
        prompt = f"Analyze this biotech news for a global professional audience in 3 sentences. Focus on the R&D impact and market significance. News: {title} - {description}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}") # 에러 로그를 남깁니다.
        return "Insight pending technical review."

url = f"https://gnews.io/api/v4/search?q=biotech+pharma&lang=en&token={GNEWS_API_KEY}"
response = requests.get(url)
news_data = response.json()

if 'articles' in news_data:
    for article in news_data['articles'][:5]:
        article['ai_analysis'] = get_ai_summary(article['title'], article['description'])

with open('news.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=4)
