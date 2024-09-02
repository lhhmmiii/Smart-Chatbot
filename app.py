import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.types import AskFileResponse, ThreadDict
from chainlit.input_widget import Select, Switch, Slider
from Process_Document import extract_word_content, extract_info
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor # The agent executor is the runtime for an agent. This is what actually calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats.
from langchain.agents.structured_chat.prompt import SUFFIX
from history_chatbot import *
from tools import *
import uuid

## -------------------------- Lấy api từ file .env(file này tôi sẽ không public nên các bạn từ tạo theo format sau nhé) ----------------------- ##
'''
HUGGINGFACEHUB_API_TOKEN = ........
GEMINI_API = .........
LANGCHAIN_API_KEY = .............
TAVILY_API_KEY = ....................
LITERAL_API_KEY = ....................
CHAINLIT_AUTH_SECRET = ..............
'''
load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')

## ---------------------- Các action của chatbot ------------------- ##
async def present_actions():
    actions = [
        cl.Action(name="Summarize document", value="summarize_document", description="Summarize document"),
    ]
    await cl.Message(content="**Hãy chọn chức năng bạn muốn thực hiện.**", actions=actions).send()


uploaded_file = None
## --------------------- Đoạn chat gọi lại phần tóm tắt file PDF được đưa vào ------------------ ##
@cl.action_callback("Summarize document")
async def on_action(action):
    memory = cl.user_session.get("memory")
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
            loader = PyPDFLoader(uploaded_file.path)
            docs = loader.load_and_split()
            cl.user_session.set("pages", loader)
            elements = [
                cl.Pdf(name="PDF file", display="inline", path=uploaded_file.path)
            ]
            await cl.Message(content="**Dưới đây là nội dung của file PDF:**", elements=elements).send()
            # try:  # Nếu nó chạy vộ lệnh try thì sẽ không có tóm tắt, chưa biết cách khắc phục nên tạm khóa lại
            #     list_page_content = extract_info(uploaded_file.path)
            #     for i, page_content in enumerate(list_page_content):
            #         content += f'**Page {i + 1}:**\n {page_content}\n'
            #     summarized_document_text = summarize_document(content)
            #     await cl.Message(content=f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
            # except Exception as e:
            #     await cl.Message(content = "Xảy ra lỗi khi xử lí nội dung PDF. Tôi sẽ chỉnh sửa sau.").send()
    
    await present_actions()
    await action.remove()
    # Lưu user session
    cl.user_session.set("content", content)
    cl.user_session.set("file_id", uploaded_file.id)
    cl.user_session.set("type", uploaded_file.type)
    cl.user_session.set("docs", docs)
    cl.user_session.set("action_type", "1")
    cl.user_session.set("is_resumne", False)
    # Lưu docs vô memory
    memory.chat_memory.add_user_message(docs)


# Đặt tên session cho mỗi đoạn chat
def create_session_id():
    session_id = str(uuid.uuid4())
    return session_id

## ------------------------ Bắt đầu chatbot -------------------------- ##
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    settings = await cl.ChatSettings(
        [
            Select(id="Model",label="Gemini - Model",
                values = ['gemini-1.5-flash', 'gemini-1.5-pro'],
                initial_index = 0,
            ),
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
    model_name = settings['Model']
    llm = ChatGoogleGenerativeAI(model = model_name, max_retries= 2, timeout= None, max_tokens = None, google_api_key=google_genai_api_key)
    session_id = create_session_id()
    # Lưu các user session
    cl.user_session.set("LLM", llm)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("action_type", "0")
    cl.user_session.set("chat_history", [])
    await present_actions()

## ---------------- Lấy nội dung từ document hoặc từ người dùng nhập vào --------------------- ##
def get_documents(text_splitter):
    is_resume = cl.user_session.get("is_resume")
    docs = None
    if not is_resume:
        file_type = cl.user_session.get("type")
        if file_type == "text/plain":
            document_text = cl.user_session.get("content")
            docs = text_splitter.create_documents([document_text])
        elif file_type == 'application/pdf':
            docs = cl.user_session.get("docs")
            cl.user_session.set("docs", docs)
    return docs

## --------------------- Tạo retriever từ document hoặc input từ người dùng. --------------------- ##
def create_retriever(docs, session_id):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key = google_genai_api_key)
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    save_path = f'Database/Document/session_{session_id}'
    vectordb.save_local(save_path)
    retriever = vectordb.as_retriever()
    return retriever

def load_vectordb(session_id):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key = google_genai_api_key)
    save_path = f'Database/Document/session_{session_id}'
    vectordb = FAISS.load_local(save_path, embedding, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    return retriever

## -------------------- Chạy khi người dùng input ----------------- ##
@cl.on_message
async def on_message(message: cl.Message):
    # Lấy các user session
    action_type = cl.user_session.get("action_type")
    llm = cl.user_session.get("LLM")
    session_id = cl.user_session.get("session_id")
    memory = cl.user_session.get("memory")
    chat_history = cl.user_session.get("chat_history")
    # await cl.Message(content = f"{action_type}_{llm}_{tools}_{memory}").send()
    # Khi chưa import document
    if action_type == "0":
        tools = my_tools(llm)
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent = agent, tools = tools)
        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        res = agent_with_chat_history.invoke(
            {"input": message.content}, 
            config= {
                "configurable": {"session_id": session_id}
                }
            )
        answer = res["output"]
        if ".png" in answer:
            elements = [
            cl.Image(
                path = f"Image/Output/{answer}",
                content = "Image created by above prompt",
                name = answer,
                display = "inline",
            )
            ]

            await cl.Message(content = answer, elements=elements).send()
        else:
            await cl.Message(content = answer).send()

    # Khi đã import document
    elif action_type == "1":
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    is_separator_regex=False,
                )
        docs = get_documents(text_splitter)
        retriever = None
        if docs == None:
            retriever = load_vectordb(session_id)
        else:
            retriever = create_retriever(docs, session_id)
        # Chain
        conversational_rag_chain = conversational_chain(llm, retriever)
        answer = conversational_rag_chain.invoke(
            {"input": message.content},
            config={
                "configurable": {"session_id": session_id}
            },
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

        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(answer)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("is_resume", True)
    settings = await cl.ChatSettings(
    [
        Select(id="Model",label="Gemini - Model",
            values = ['gemini-1.5-flash', 'gemini-1.5-flash'],
            initial_index = 0,
        ),
        Slider(id = "Temperature", label = "temperature", initial = 1, min = 0, max = 1, step = 0.05),
        Slider(id="Top-k",label = "Top-k", initial = 1, min = 1, max = 100, step = 1),
        Slider(id="Top-p",label = "Top-p",initial = 1, min = 0,max = 1,step = 0.02),
    ]).send()
    model_name = settings['Model']
    llm = ChatGoogleGenerativeAI(model = model_name, max_retries= 2, timeout= None, max_tokens = None, google_api_key=google_genai_api_key)
    cl.user_session.set("LLM",llm)
    #
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)


@cl.password_auth_callback 
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("LHH", "1323"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None