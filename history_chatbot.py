import os
import hashlib
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage


load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

def question_answering_chain(llm):
    # Template prompt này dùng để trả lời các câu hỏi liên quan tới document
    template_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    \n\n
    {context}
    """

    prompt1 = ChatPromptTemplate.from_messages(
        [
            ("system", template_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, prompt1)
    return qa_chain


def chat_retriever_chain(llm, retriever):
    # Template prompt này dùng để liên kết câu hỏi sau với lịch sử trò chuyện.
    '''
    Ví dụ:
    Human: Machine Learning là gì?
    System: answer
    Human: Nó bao gồm bao nhiêu loại? -> Nó ở đây là machine learning nhưng chúng ta phải biết câu hỏi trước thì ta mới xác định được. Vậy nên prompt này sẽ giúp ích cho việc đó.
    Nó sẽ giúp ta truy vấn ngược lại lịch sử trò chuyện để xác định "Nó" là gì?
    '''
    template_prompt2 = """ 
    Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    
    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", template_prompt2),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, prompt2
    )
    '''
    Create a chain that takes conversation history and returns documents.
    If there is no chat_history, then the input is just passed directly to the retriever. If there is chat_history, then the prompt and LLM will be used to generate a search query.
    That search query is then passed to the retriever.
    '''
    return history_aware_retriever


history_store = {}
def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]


def conversational_chain(llm, retriever):
    qa_chain = question_answering_chain(llm)
    history_aware_retriever = chat_retriever_chain(llm, retriever)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain



# conversational_rag_chain.invoke(
#     {"input": "What is Task Decomposition?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]

# # ------------------------------------------------------------------------ #
# import os
# from dotenv import load_dotenv
# import chainlit as cl
# from chainlit.types import AskFileResponse, ThreadDict
# from typing import Optional, Dict, List
# from chainlit.input_widget import Select, Switch, Slider
# from Process_Document import extract_word_content, extract_info
# from document_summarize import *
# from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableConfig
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import WebBaseLoader

# google_genai_api_key = os.getenv('GEMINI_API')
# llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', max_retries= 2, timeout= None, max_tokens = None, api_key=google_genai_api_key)
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_genai_api_key)
# vectordb = FAISS.from_documents(docs, embedding=embedding)
# retriever = vectordb.as_retriever()

# # ------------------------------------------------------------------------ #