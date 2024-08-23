import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.input_widget import Select, Switch, Slider
from Process_Document import extract_word_content, extract_info, extract_content_from_web
from document_summarize import *
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = pinecone_api_key

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Document is use only vietnamese language. So, you use it to reply.
Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

uploaded_file = None
collection_name = set()

## ---------------------- Các action của chatbot ------------------- ##
async def present_actions():
    actions = [
        cl.Action(name="Summarize document", value="summarize_document", description="Summarize document"),
        cl.Action(name="Summarize content from web", value="summarize content from web", description="Summarize content from web"),
        cl.Action(name="Summarize a text you input", value="Summarize a text you input", description="Summarize text you input"),
    ]
    await cl.Message(content="**Hãy chọn chức năng bạn muốn thực hiện.**", actions=actions).send()


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

## --------------------------- Nút gọi lại của tóm tắt nội dung từ trang web ------------------------------------- ##
@cl.action_callback("Summarize content from web")
async def on_action(action):
    res = await cl.AskUserMessage(content = "**Nhập đường dẫn tới trang web:**", timeout=30).send()
    if res:
        # Loader
        loader = AsyncChromiumLoader([res['output']])
        html = loader.load()
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            html # tags_to_extract=["p", "li", "div", "a"]
        )
        # Content
        content = extract_content_from_web(res['output'])
        await cl.Message(
            content = f"**Content:**\n {content}"
        ).send()

    summarized_document_text = summarize_document(content)
    await cl.Message(content = f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
    await action.remove()
    await present_actions()
    cl.user_session.set("content", content)
    cl.user_session.set("action_type", "2")
    cl.user_session.set("html_docs", docs_transformed)

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
    cl.user_session.set("action_type", "3")


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
        content="Chatbot có nhiệm vụ tóm tắt document và trả lời tất cả câu hỏi liên quan tới document bạn cung cấp.\nHiện tại chatbot chỉ tóm tắt các nội dung bằng Tiếng Việt.",
        elements=[image],
    ).send()

    llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', max_retries= 2, timeout= None, max_tokens = None, api_key=google_genai_api_key)
    cl.user_session.set("LLM", llm)

    await present_actions()



@cl.on_message
async def on_message(message: cl.Message):
    llm = cl.user_session.get("LLM")
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
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
        docs = cl.user_session.get("html_docs")
    elif action_type == "3":
        document_text = cl.user_session.get("content")
        docs = text_splitter.create_documents([document_text])
    # Embedding content
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_genai_api_key)
    vectordb = FAISS.from_documents(docs, embedding = embedding)
    retriever = vectordb.as_retriever()
    # Prompt
    prompt = PromptTemplate.from_template(template)
    chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
     }
    | prompt
    | llm
    | StrOutputParser()
    )

    msg = cl.Message(content="")

    async for chunk in chain.astream(message.content):
        await msg.stream_token(chunk)

    await msg.send()

