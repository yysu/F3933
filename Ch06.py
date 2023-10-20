import getpass
import openai
import tiktoken
import yfinance as yf
import numpy as np
import requests
import datetime as dt
from bs4 import BeautifulSoup
import pandas as pd
class StockInfo():
  # 取得全部股票的股號、股名
  def stock_name(self):
    # print("線上讀取股號、股名、及產業別")
    response = requests.get('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2')
    url_data = BeautifulSoup(response.text, 'html.parser')
    stock_company = url_data.find_all('tr')
  
    # 資料處理
    data = [
        (row.find_all('td')[0].text.split('\u3000')[0].strip(),
          row.find_all('td')[0].text.split('\u3000')[1],
          row.find_all('td')[4].text.strip())
        for row in stock_company[2:] if len(row.find_all('td')[0].text.split('\u3000')[0].strip()) == 4
    ]
  
    df = pd.DataFrame(data, columns=['股號', '股名', '產業別'])
  
    return df
  # 取得股票名稱
  def get_stock_name(self, stock_id, name_df):
      return name_df.set_index('股號').loc[stock_id, '股名']

class StockAnalysis():
  def __init__(self,openai_api_key):
    # 初始化 OpenAI API 金鑰
    openai.api_key = openai_api_key
    # self.openai_api_key = getpass.getpass("請輸入金鑰：")  # 請在使用時設定 API 金鑰
    self.stock_info = StockInfo()  # 實例化 StockInfo 類別
    self.name_df = self.stock_info.stock_name()
  # 從 yfinance 取得一周股價資料
  def stock_price(self, stock_id="大盤", days = 15):
    if stock_id == "大盤":
      stock_id="^TWII"
    else:
      stock_id += ".TW"
  
    end = dt.date.today() # 資料結束時間
    start = end - dt.timedelta(days=days) # 資料開始時間
    # 下載資料
    df = yf.download(stock_id, start=start)
  
    # 更換列名
    df.columns = ['開盤價', '最高價', '最低價',
                  '收盤價', '調整後收盤價', '成交量']
  
    data = {
      '日期': df.index.strftime('%Y-%m-%d').tolist(),
      '收盤價': df['收盤價'].tolist(),
      '每日報酬': df['收盤價'].pct_change().tolist(),
      # '漲跌價差': df['調整後收盤價'].diff().tolist()
      }
  
    return data
  # 基本面資料
  def stock_fundamental(self, stock_id= "大盤"):
    if stock_id == "大盤":
        return None
  
    stock_id += ".TW"
    stock = yf.Ticker(stock_id)
  
    # 營收成長率
    quarterly_revenue_growth = np.round(stock.quarterly_financials.loc["Total Revenue"].pct_change(-1).dropna().tolist(), 2)
  
    # 每季EPS
    quarterly_eps = np.round(stock.quarterly_financials.loc["Basic EPS"].dropna().tolist(), 2)
  
    # EPS季增率
    quarterly_eps_growth = np.round(stock.quarterly_financials.loc["Basic EPS"].pct_change(-1).dropna().tolist(), 2)
  
    # 轉換日期
    dates = [date.strftime('%Y-%m-%d') for date in stock.quarterly_financials.columns]
  
    data = {
        '季日期': dates[:len(quarterly_revenue_growth)],  # 以最短的數據列表長度為准，確保數據對齊
        '營收成長率': quarterly_revenue_growth.tolist(),
        'EPS': quarterly_eps.tolist(),
        'EPS 季增率': quarterly_eps_growth.tolist()
    }
  
    return data
  # 新聞資料
  def stock_news(self, stock_name ="大盤"):
    if stock_name == "大盤":
      stock_name="台股 -盤中速報"
  
    data=[]
    # 取得 Json 格式資料
    json_data = requests.get(f'https://ess.api.cnyes.com/ess/api/v1/news/keyword?q={stock_name}&limit=6&page=1').json()
  
    # 依照格式擷取資料
    items=json_data['data']['items']
    for item in items:
        # 網址、標題和日期
        news_id = item["newsId"]
        title = item["title"]
        publish_at = item["publishAt"]
        # 使用 UTC 時間格式
        utc_time = dt.datetime.utcfromtimestamp(publish_at)
        formatted_date = utc_time.strftime('%Y-%m-%d')
        # 前往網址擷取內容
        url = requests.get(f'https://news.cnyes.com/'
                          f'news/id/{news_id}').content
        soup = BeautifulSoup(url, 'html.parser')
        p_elements=soup .find_all('p')
        # 提取段落内容
        p=''
        for paragraph in p_elements[4:]:
            p+=paragraph.get_text()
        data.append([stock_name, formatted_date ,title,p])
    return data
    
  # 建立 GPT 3.5-16k 模型
  def get_reply(self, messages):
    try:
      response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-16k",
          temperature=0,
          messages=messages
      )
      reply = response["choices"][0]["message"]["content"]
    except openai.OpenAIError as err:
      reply = f"發生 {err.error.type} 錯誤\n{err.error.message}"
    return reply
  
  # 設定 AI 角色, 使其依據使用者需求進行 df 處理
  def ai_helper(self, table_name, user_msg):
    msg = [
    {
      "role": "system",
      "content":
      f"我只需要一個名為 'calculate(df)' 的 Python 函式來解答問題{user_msg}, 請注意問題中的時間線和關鍵字,"\
      "如有時間請以現在時間 (now) 為主,不是資料時間。"\
      f"我會提供表格的欄位 {table_name},請使用我給的欄位作為 DataFrame 的欄位,請勿自己生成欄位, 我給的欄位都有資料。"
      "表格資料結構為多檔股票不同時間的數據, 如果需要可以使用 DataFrame 的 'groupby'和'pct_change' 函數。"\
      "必須過濾 NaN ,不需要完整資料, 只要跟問題有關的資料即可。"\
      "最後函式返回一個 df 且必須是 DataFrame 格式, 請確保在函式最後將其轉換為 DataFrame。請只使用 pandas 進行計算。"\
      "Please note that your response should solely consist of the code itself, and no additional information should be included. "
    }, 
    {
      "role":
      "user",
      "content":
      "請確保生成的程式碼僅為 'calculate' 函式，不包括其他無關的內容或輸出。"\
      "Please note that your response should solely consist of the code itself, and no additional information should be included. "
    }]
    
    
    reply_data = self.get_reply(msg)
    return reply_data
  
  # 建立訊息指令(Prompt)
  def generate_content_msg(self, stock_id, name_df):
  
      stock_name = self.stock_info.get_stock_name(
          stock_id, name_df) if stock_id != "大盤" else stock_id
  
      price_data = self.stock_price(stock_id)
      news_data = self.stock_news(stock_name)
  
      content_msg = f'你現在是一位專業的證券分析師, '\
        '你會依據以下資料來進行分析並給出一份完整的分析報告:\n'
  
      content_msg += f'近期價格資訊:\n {price_data}\n'
  
      if stock_id != "大盤":
          stock_value_data = self.stock_fundamental(stock_id)
          content_msg += f'每季營收資訊：\n {stock_value_data}\n'
  
      content_msg += f'近期新聞資訊: \n {news_data}\n'
      content_msg += f'請給我{stock_name}近期的趨勢報告,請以詳細、'\
        '嚴謹及專業的角度撰寫此報告,並提及重要的數字, reply in 繁體中文'
  
      return content_msg

  # StockGPT
  def stock_gpt(self, stock_id):
      content_msg = self.generate_content_msg(stock_id, self.name_df)
      msg = [{
          "role": "system",
          "content": f"你現在是一位專業的證券分析師, 你會統整近期的股價漲幅"\
        "、基本面、新聞資訊等方面並進行分析, 然後生成一份專業的趨勢分析報告, tokens數量上限為1600"
      }, {
          "role": "user",
          "content": content_msg
      }]
  
      reply_data = self.get_reply(msg)
  
      return reply_data
