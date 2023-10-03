import os
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
    def pdf_loader(self,file,size,overlap):
        loader = PDFPlumberLoader(file)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                                chunk_overlap=overlap)
        new_doc = text_splitter.split_documents(doc)
        db = FAISS.from_documents(doc, OpenAIEmbeddings())
        db.save_local(f'/content/drive/MyDrive/DB/{file}')
        return db
    def analyze_chain(self,db,input):
        data = db.max_marginal_relevance_search(input)
        word = self.word_chain.run(input)
        result = self.data_chain({'output':word,'text':data})
        return result['text']
