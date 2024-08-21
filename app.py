import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from Process_Document import extract_word_content, extract_info, extract_content_from_web
from document_summarize import *

uploaded_file = None

async def present_actions():
    actions = [
        cl.Action(name="Summarize document", value="summarize_document", description="Summarize document"),
        cl.Action(name="Summarize content from web", value="summarize content from web", description="Summarize content from web"),
        cl.Action(name="Summarize a text you input", value="Summarize a text you input", description="Summarize text you input"),
        cl.Action(name="QA with chatbot", value="QA with chatbot", description='QA with chatbot about content you provide')
    ]
    await cl.Message(content="**Hãy chọn chức năng bạn muốn thực hiện.**", actions=actions).send()

@cl.action_callback("Summarize document")
async def on_action(action):
    # Yêu câu người dùng đưa document vào
    global uploaded_file
    is_reuse = True
    if uploaded_file is not None:
        actions = [
            cl.Action(name = "Yes", value = "1", label='Có'),
            cl.Action(name = "No", value = "0", label = "Không")
        ]
        res = await cl.AskActionMessage(
            content=f"Bạn có muốn sử dụng document khác",
            actions= actions
        ).send()
        if res.get("value") == "1":
            is_reuse = False
    if uploaded_file is None or not is_reuse:
        files = None
        while files is None:
            files = await cl.AskFileMessage(content="Vui lòng đưa document vào (PDF, Word, TXT)", accept=["text/plain", "application/pdf"], timeout=180, max_size_mb=20).send()
        uploaded_file = files[0]
    
        content = ""
        if uploaded_file.type == "text/plain":
            with open(uploaded_file.path, "r", encoding="utf-8") as f:
                content = f.read()
        elif uploaded_file.type == "application/pdf":
            elements = [
                cl.Pdf(name="PDF file", display="inline", path=uploaded_file.path)
            ]
            await cl.Message(content="Đưới đây là nội dung của file PDF.", elements=elements).send()
            list_page_content = extract_info(uploaded_file.path)
            for i, page_content in enumerate(list_page_content):
                content += f'**Page {i + 1}:**\n {page_content}\n'

        summarized_document_text = summarize_document(content)
        await cl.Message(content=f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
        await present_actions()

@cl.action_callback("Summarize content from web")
async def on_action(action):
    res = await cl.AskUserMessage(content = "Nhập đường dẫn tới trang web: ", timeout=30).send()
    if res:
        await cl.Message(
            content=f"**Link:** {res['output']}",
        ).send()
        content = extract_content_from_web(res['output'])
        await cl.Message(
            content = f"**Content:**\n {content}"
        ).send()

    summarized_document_text = summarize_document(content)
    await cl.Message(content = f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
    await action.remove()
    await present_actions()

@cl.action_callback("Summarize a text you input")
async def on_action(action):
    res = await cl.AskUserMessage(content = "Nhập nội dung bạn muốn tóm tắt: ", timeout=30).send()
    if res:
        await cl.Message(
            content=f"**Content:**\n {res['output']}",
        ).send()

    summarized_document_text = summarize_document(res['output'])
    await cl.Message(content = f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()
    await action.remove()
    await present_actions()

@cl.action_callback("QA with chatbot")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()

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

    await present_actions()


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content
    await present_actions()

