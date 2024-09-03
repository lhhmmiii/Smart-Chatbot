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
from generate_related_question import *

## -------------------------- Lấy api từ file .env(file này tôi sẽ không public nên các bạn từ tạo theo format sau nhé) ----------------------- ##
'''
HUGGINGFACEHUB_API_TOKEN =  .....................
GEMINI_API =  .....................
LANGCHAIN_API_KEY = .....................
TAVILY_API_KEY =  .....................
LITERAL_API_KEY =  .....................
CHAINLIT_AUTH_SECRET =  .....................
STABILITY_KEY =  .....................
'''
load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## ---------------------- Các action của chatbot ------------------- ##
async def present_actions():
    actions = [
        cl.Action(name="Document QA", value="Document QA", description="Document question answering"),
    ]
    await cl.Message(content="**Hãy chọn chức năng bạn muốn thực hiện.**", actions=actions).send()

# Create from document that user upload
def create_retriever(docs, session_id):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key = google_genai_api_key)
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    save_path = f'Database/Document/session_{session_id}'
    vectordb.save_local(save_path)
    retriever = vectordb.as_retriever()
    return retriever


uploaded_file = None
## --------------------- Đoạn chat gọi lại phần tóm tắt file PDF được đưa vào ------------------ ##
@cl.action_callback("Document QA")
async def on_action(action):
    content = ""
    docs = None
    files = None
    while files is None:
        files = await cl.AskFileMessage(content="Vui lòng đưa document vào (PDF, Word, TXT)", accept=["text/plain", "application/pdf"], timeout=180, max_size_mb=20).send()
    
    uploaded_file = files[0]

    if uploaded_file.type == "text/plain":
        with open(uploaded_file.path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = text_splitter.create_documents([content])
        await cl.Message(content=f"**Dưới đây là nội dung của file text:**\n{content}").send()
    elif uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file.path)
        docs = loader.load_and_split()
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
    
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
    session_id = cl.user_session.get("session_id")
    retriever = create_retriever(docs, session_id)
    # Generate questions related the document
    generate_questions_chain = generate_question_runnables(llm, retriever)
    msg = cl.Message(content = "")
    async for chunk in generate_questions_chain.astream("Vietnamese"):
        await msg.stream_token(chunk)
    await msg.send()
    await action.remove()
    # Lưu user session
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("action_type", "1")

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
        content="**Các chức năng của chatbot**\n - Trả lời những câu hỏi trong tầm kiến thức của gemini \n - Tìm kiếm những thông tin bạn muốn(bổ sung cho sự thiếu sót của LLM)\n- Trả lời tất cả câu hỏi liên quan tới document bạn cung cấp.\n - Tóm tắt nội dung bạn cung cấp bằng Tiếng Việt.\n- Sinh ảnh và chỉnh sửa ảnh\n",
        elements=[image],
    ).send()
    model_name = settings['Model']
    llm = ChatGoogleGenerativeAI(model = model_name, max_retries= 2, timeout= None, max_tokens = None, google_api_key=google_genai_api_key)
    session_id = create_session_id()
    # Lưu các user session
    cl.user_session.set("LLM", llm)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("action_type", "0")
    await present_actions()


## Load vectordb from local

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
    is_resume = cl.user_session.get("is_resume")
    # Khi chưa upload document
    if action_type == "0":
        tools = my_tools(llm)
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent = agent, tools = tools, verbose=True)
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
        save_history_store()
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
        retriever = None
        if is_resume:
            retriever = load_vectordb(session_id)
        else:
            retriever = cl.user_session.get("retriever")
        # Document QA chain
        conversational_rag_chain = conversational_chain(llm, retriever)
        answer = conversational_rag_chain.invoke(
            {"input": message.content},
            config={
                "configurable": {"session_id": session_id}
            },
        )["answer"]

        await cl.Message(content = answer).send()

        # Lưu lịch sửa hội thoại
        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(answer)

# @cl.on_chat_resume
# async def on_chat_resume(thread: ThreadDict):
#     cl.user_session.set("is_resume", True)
#     settings = await cl.ChatSettings(
#     [
#         Select(id="Model",label="Gemini - Model",
#             values = ['gemini-1.5-flash', 'gemini-1.5-flash'],
#             initial_index = 0,
#         ),
#         Slider(id = "Temperature", label = "temperature", initial = 1, min = 0, max = 1, step = 0.05),
#         Slider(id="Top-k",label = "Top-k", initial = 1, min = 1, max = 100, step = 1),
#         Slider(id="Top-p",label = "Top-p",initial = 1, min = 0,max = 1,step = 0.02),
#     ]).send()
#     model_name = settings['Model']
#     llm = ChatGoogleGenerativeAI(model = model_name, max_retries= 2, timeout= None, max_tokens = None, google_api_key=google_genai_api_key)
#     cl.user_session.set("LLM", llm)    
#     #
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