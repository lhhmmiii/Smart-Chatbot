import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from Process_Document import extract_word_content
from document_summarize import *


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
        content="Chatbot có nhiệm vụ tóm tắt document và trả lời tất cả câu hỏi liên quan tới document bạn cung cấp",
        elements=[image],
    ).send()

    # Yêu câu người dùng đưa document vào
    files = None
    while files == None:
        files = await cl.AskFileMessage(content = "Vui lòng đưa document vào(PDF, Word, TXT)", accept=["text/plain"],timeout = 180).send()
        text_file = files[0]
        with open(text_file.path, "r", encoding="utf-8") as f:
            text = f.read()  

    await cl.Message(content = f"**Content:**\n {text}").send()
    summarized_document_text = summarize_document(text)
    await cl.Message(content = f"**Nội dung tóm tắt của document:**\n {summarized_document_text}").send()

@cl.on_message
async def on_message(message: cl.Message):
    content = message.content

