import os
import time
import random
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
class PdfLoader:
    def __init__(self,openai_api_key):
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        self.data_prompt=ChatPromptTemplate.from_messages(messages=[
            ("system","你現在是一位專業的證券分析師,"
            "你會統整年報並進行分析, 針對{output}作分析, 然後生成一份專業的趨勢分析報告。"),
            ("human","{text}")])
        self.data_chain = LLMChain(llm=self.llm, prompt=self.data_prompt)
        self.word_prompt=ChatPromptTemplate.from_messages(messages=[
            ("system","你可以將使用者輸入的句子取出一個關鍵字,"
            "要取出的關鍵字會是以年報中的會出現的相關名詞為主"),
            ("human","{input}")])
        self.word_chain = LLMChain(llm=self.llm, prompt=self.word_prompt)
    def annual_report(self,id,y):
        wait_time = random.uniform(2,6)
        url = 'https://doc.twse.com.tw/server-java/t57sb01'
        # 建立 POST 請求的表單
        data = {
            "id":"",
            "key":"",
            "step":"1",
            "co_id":id,
            "year":y,
            "seamon":"",
            "mtype":'F',
            "dtype":'F04'
        }
        # 發送 POST 請求
        with requests.post(url, data=data) as response:
            time.sleep(wait_time)
            # 取得回應後擷取檔案名稱
            link=BeautifulSoup(response.text, 'html.parser')
            link1=link.find('a').text
            print(link1)
    
        # 建立第二個 POST 請求的表單
        data2 = {
            'step':'9',
            'kind':'F',
            'co_id':id,
            'filename':link1 # 檔案名稱
        }
        # 發送 POST 請求
        with requests.post(url, data=data2) as response2:
            time.sleep(wait_time)
            link=BeautifulSoup(response2.text, 'html.parser')
            link1=link.find('a')['href']
            print(link1)
    
        # # 發送 GET 請求
        response3 = requests.get('https://doc.twse.com.tw' + link1)
        time.sleep(wait_time)
        # 取得 PDF 資料
        folder_path = '/content/drive/MyDrive/StockGPT/PDF/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + y + '_' + id + '.pdf', 'wb') as file:
            file.write(response3.content)
        print('OK')
    def pdf_loader(self,file,size,overlap):
        loader = PDFPlumberLoader(file)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                                chunk_overlap=overlap)
        new_doc = text_splitter.split_documents(doc)
        db = FAISS.from_documents(new_doc, OpenAIEmbeddings())
        file_name = file.split("/")[-1].split(".")[0]
        db_file = '/content/drive/MyDrive/StockGPT/DB/'
        if not os.path.exists(db_file):
            os.makedirs(db_file)
        db.save_local(db_file + file_name)
        return db
    def analyze_chain(self,db,input):
        data = db.max_marginal_relevance_search(input)
        word = self.word_chain.run(input)
        result = self.data_chain({'output':word,'text':data})
        return result['text']
