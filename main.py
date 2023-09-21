import getpass
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
class pdfai():
os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "你是一個根據年報資料與上下文作回答的助手,"
            "有明確數據可以用數據回答,"
            "回答以繁體中文和台灣用語為主。"
            "{context}"
            "{history}"
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    )
memory = ConversationBufferWindowMemory(k=2, return_messages=True,input_key='question')
chain_type_kwargs = {'prompt':prompt, 'memory':memory}
def pdf_loader(file,size,overlap):
    loader =  PyPDFLoader(file)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    new_doc = text_splitter.split_documents(doc)
    db = FAISS.from_documents(new_doc, OpenAIEmbeddings())
    return db
def question_and_answer(retriever,q):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs)
    result = qa(q)
    return result
