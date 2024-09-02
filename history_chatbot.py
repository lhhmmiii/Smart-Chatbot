import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import chainlit as cl


load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# ------------------------------ Conversional Chatbot when it have retriever(document) ------------------------------- #
def question_answering_chain(llm):
    # Template prompt này dùng để trả lời các câu hỏi liên quan tới document
    template_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    \n\n
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
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
    cl.user_session.set("history_session", history_store[session_id])
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