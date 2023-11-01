import getpass
import openai
import tiktoken
import yfinance as yf
import numpy as np
import requests
import os
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
  def ai_helper(self, df_company, df_daily, df_quarterly, user_msg):
      
      code_example ='''
  def calculate(df_company, df_daily, df_quarterly):
    
    # 找出最近兩個季度的日期
    latest_two_dates = df_quarterly['日期'].drop_duplicates().sort_values(ascending=False).head(2)
    
    # 選出最近兩季的資料
    recent_two_quarters_data = df_quarterly[df_quarterly['日期'].isin(latest_two_dates)]
    
    # 從 df_quarterly 中計算每支股票的平均營業收入
    recent_two_quarters_data['營業收入成長率'] = recent_two_quarters_data.groupby('股號')['營業收入'].pct_change()
    recent_two_quarters_data.dropna(subset=['營業收入成長率'], inplace=True)
    # 將平均營業收入與 df_company 合併，然後選出半導體業中平均營業收入最高的10檔股票
    df_company_with_growth_rate = pd.merge(df_company, recent_two_quarters_data[['股號', '營業收入成長率']], on='股號', how='left')
    
    semiconductor_stocks = df_company_with_growth_rate[df_company_with_growth_rate['產業別'] == '半導體業'].sort_values(by='營業收入成長率', ascending=False).head(10)
  
    return semiconductor_stocks
      '''
  
      user_requirement = [{
          "role": "user",
          "content": f"The user requirement:{user_msg}\n\
          df_company.columns ={df_company.columns}\n\
          df_daily.columns ={df_daily.columns}\n\
          df_quarterly.columns ={df_quarterly.columns}\n\
          Your task is to develop a Python function named \
          'calculate(df_company, df_daily, df_quarterly)' and return a new dataframe based on df_company."
      }]
  
      msg = [{
        "role":
        "system",
        "content":
        f"You will act as a professional Python code generation robot. \
          Based on user requirements, you will manipulate the company data\
          in the df table and perform stock selection. I will provide \
          you with 3 df tables: df_compaany, df_daily, and df_quarterly. \n\
          Please note that your response should solely \
          consist of the code itself, \
          and no additional information should be included."
      }, {
        "role":
        "user",
        "content":
        f"The user requirement:請選出半導體業且近期平均營收最高的10檔股票 \n\
          Your task is to develop a Python function named \
          'calculate(df_company, df_daily, df_quarterly)'. Ensure that you only utilize the columns \
          present in the dataset, \n\
          The df_company table contains basic company information with columns:{df_company.columns}\n\
          The df_daily table is a daily stock price table with columns:{df_daily.columns}\n\
          The df_quarterly table is a quarterly revenue table with columns:{df_quarterly.columns}\n\
          After processing, the function should return a new DataFrame based on df_company."
      }, {
        "role":
        "assistant",
        "content":f"{code_example}"
      }]
      msg += user_requirement
  
  
      reply_data = self.get_reply(msg)
      return user_requirement, reply_data

  def ai_debug(self, history, code_str ,error_msg):
      msg = [{
          "role": "system",
          "content":
          "You will act as a professional Python code generation robot. \
          I will send you the incorrect code and error message.\
          Please correct and return the fixed code. \n\
          Please note that your response should solely \
          consist of the code itself, \
          and no additional information should be included."}]
      msg += history
      msg += [{
          "role": "system",
          "content":f"{code_str}"
      }, {
          "role": "user",
          "content": f"The error code:{code_str} \n\
          The error message:{error_msg} \n\
          Please reconfirm user requirements \n\
          Your task is to develop a Python function named \
          'calculate(df_company, df_daily, df_quarterly)', \
          Please note that your response should solely \
          consist of the code itself, \
          and no additional information should be included."
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
