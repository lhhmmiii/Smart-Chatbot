import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.types import AskFileResponse, ThreadDict
from typing import Optional, Dict, List
from chainlit.input_widget import Select, Switch, Slider
from Process_Document import extract_word_content, extract_info
from document_summarize import *
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from history_chatbot import *
from agent import *


## -------------------------- Lấy api từ file .env(file này tôi sẽ không public nên các bạn từ tạo theo format sau nhé) ----------------------- ##
'''
HUGGINGFACEHUB_API_TOKEN = ........
GEMINI_API = .........
LANGCHAIN_API_KEY = .............
'''
load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

## ---------------------- Format lại output ------------------- ##
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


## ---------------------- Các action của chatbot ------------------- ##
async def present_actions():
    actions = [
        cl.Action(name="Summarize document", value="summarize_document", description="Summarize document"),
        cl.Action(name="Summarize a text you input", value="Summarize a text you input", description="Summarize text you input"),
    ]
    await cl.Message(content="**Hãy chọn chức năng bạn muốn thực hiện.**", actions=actions).send()


uploaded_file = None
## --------------------- Đoạn chat gọi lại phần tóm tắt file PDF được đưa vào ------------------ ##
@cl.action_callback("Summarize document")
async def on_action(action):
    global uploaded_file
    is_reuse = True
    
    # Check if a document has already been uploaded
    if uploaded_file is not None:
        actions = [
            cl.Action(name="Yes", value="1", label='Có'),
            cl.Action(name="No", value="0", label="Không")
        ]
        res = await cl.AskActionMessage(
            content="Bạn có muốn sử dụng document khác?",
            actions=actions
        ).send()
        if res.get("value") == "1":
            is_reuse = False

    content = ""
    docs = None
    if uploaded_file is None or not is_reuse:
        await cl.Message(content="").send()
        
        files = None
        while files is None:
            files = await cl.AskFileMessage(content="Vui lòng đưa document vào (PDF, Word, TXT)", accept=["text/plain", "application/pdf"], timeout=180, max_size_mb=20).send()
        
        uploaded_file = files[0]
        
        if uploaded_file.type == "text/plain":
            with open(uploaded_file.path, "r", encoding="utf-8") as f:
                content = f.read()
            await cl.Message(content=f"**Dưới đây là nội dung của file text:**\n{content}").send()
        elif uploaded_file.type == "application/pdf":
            loader = PyPDFLoader('Document/PDF/7698_Cac-moc-thoi-gian-KLTN-TTTN-TTDATN-K2021-SV.pdf')
            docs = loader.load_and_split()
            cl.user_session.set("pages", loader)
            elements = [
                cl.Pdf(name="PDF file", display="inline", path=uploaded_file.path)
            ]
            await cl.Message(content="**Dưới đây là nội dung của file PDF:**", elements=elements).send()
            list_page_content = extract_info(uploaded_file.path)
            for i, page_content in enumerate(list_page_content):
                content += f'**Page {i + 1}:**\n {page_content}\n'
        summarized_document_text = summarize_document(content)
        await cl.Message(content=f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
    
    await present_actions()
    await action.remove()
    cl.user_session.set("content", content)
    cl.user_session.set("file_id", uploaded_file.id)
    cl.user_session.set("type", uploaded_file.type)
    cl.user_session.set("docs", docs)
    cl.user_session.set("action_type", "1")


## --------------------------- Nút gọi lại của tóm tắt nội dung mà người dùng nhập vào------------------------------------- ##
@cl.action_callback("Summarize a text you input")
async def on_action(action):
    res = await cl.AskUserMessage(content = "**Nhập nội dung bạn muốn tóm tắt:**", timeout=30).send()
    if res:
        await cl.Message(
            content=f"**Content:**\n {res['output']}",
        ).send()

    summarized_document_text = summarize_document(res['output'])
    await cl.Message(content = f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
    await action.remove()
    await present_actions()
    cl.user_session.set("content", res['output'])
    cl.user_session.set("action_type", "2")

## ------------------------ Bắt đầu chatbot -------------------------- ##
@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Slider(id = "Temperature", label = "temperature", initial = 1, min = 0, max = 1, step = 0.05),
            Slider(id="Top-k",label = "Top-k", initial = 1, min = 1, max = 100, step = 1),
            Slider(id="Top-p",label = "Top-p",initial = 1, min = 0,max = 1,step = 0.02),
        ]
    ).send()

    # Ảnh bìa
    image = cl.Image(path = 'Image/chatbot.png', name = 'cover_image', display = 'inline')
    await cl.Message(
        content="**- Chatbot có nhiệm vụ tóm tắt document và trả lời tất cả câu hỏi liên quan tới document bạn cung cấp.**\n**- Hiện tại chatbot chỉ tóm tắt các nội dung bằng Tiếng Việt.**\n**- Nếu bạn muốn hỏi về các vấn đề bên ngoài document thì vẫn cứ hỏi nhưng tôi chưa chắc thông tin đó sẽ chính xác(đừng upload document or context trước khi hỏi).**",
        elements=[image],
    ).send()

    llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', max_retries= 2, timeout= None, max_tokens = None, api_key=google_genai_api_key)
    session_id = "ss1"
    cl.user_session.set("LLM", llm)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("action_type", "0")

    await present_actions()

## ---------------- Lấy nội dung từ document hoặc từ người dùng nhập vào --------------------- ##
def get_documents_based_on_action(text_splitter):
    action_type = cl.user_session.get("action_type")
    docs = None
    if action_type == "1":
        file_type = cl.user_session.get("type")
        if file_type == "text/plain":
            document_text = cl.user_session.get("content")
            docs = text_splitter.create_documents([document_text])
        elif file_type == 'application/pdf':
            docs = cl.user_session.get("docs")
    elif action_type == "2":
        document_text = cl.user_session.get("content")
        docs = text_splitter.create_documents([document_text])
    return docs


## --------------------- Tạo retriever từ document hoặc input từ người dùng. --------------------- ##
def create_retriever(docs):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_genai_api_key)
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    return vectordb.as_retriever()

# Chain trả lời câu hỏi khi người dùng chưa đưa document hay input content vào
def create_chain(llm):
    template = """
    Hello! I'm your assistant. How can I assist you today? When you're ready, feel free to upload the document you'd like help with. If you have any other questions before that, don't hesitate to ask!
    Question: {question} 
    You use vietnamese to reply.
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    return (
        {
            'question': RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

## -------------------- Chạy khi người dùng input ----------------- ##
@cl.on_message
async def on_message(message: cl.Message):
    action_type = cl.user_session.get("action_type")
    llm = cl.user_session.get("LLM")
    session_id = cl.user_session.get("session_id")
    if action_type == "0":
        chain = create_chain(llm)
        msg = cl.Message("")
        async for chunk in chain.astream(message.content):
            await msg.stream_token(chunk)
        
        await msg.send()
    else:
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    is_separator_regex=False,
                )
        docs = get_documents_based_on_action(text_splitter)
        # Embedding content
        retriever = create_retriever(docs)
        # Chain
        conversational_rag_chain = conversational_chain(llm, retriever)

        answer = conversational_rag_chain.invoke(
            {"input": message.content},
            config={
                "configurable": {"session_id": session_id}
            },  # constructs a key "abc123" in `store`.
        )["answer"]

        await cl.Message(content = answer).send()

        # msg = cl.Message(content="")

        # async for chunk in conversational_rag_chain.astream(
        #     {"input": message.content},
        #     config={
        #         "configurable": {"session_id": session_id}
        #     },
        # ): # Tại đây sử dùng stream để tăng trải nghiệm người dùng
        #         await msg.stream_token(chunk['answer'])

        # await msg.send()



# @cl.on_chat_resume
# async def on_chat_resume(thread: ThreadDict):
#     memory = ConversationBufferMemory(return_messages=True)
#     root_messages = [m for m in thread["steps"] if m["parentId"] == None]
#     for message in root_messages:
#         if message["type"] == "user_message":
#             memory.chat_memory.add_user_message(message["output"])
#         else:
#             memory.chat_memory.add_ai_message(message["output"])

#     cl.user_session.set("memory", memory)

# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Fetch the user matching username from your database
#     # and compare the hashed password with the value stored in the database
#     if (username, password) == ("LHH", "1323"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None