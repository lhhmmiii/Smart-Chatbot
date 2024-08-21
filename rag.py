import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from Process_Document import extract_word_content

load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = langchain_api_key 

# LLM
llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', max_retries= 2, timeout= None, max_tokens = None, api_key=google_genai_api_key)

# Document loader
list_section_content = extract_word_content('Document\Word\cccd.docx')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents(list_section_content)

# Vectore store
embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_genai_api_key)
vectordb = Chroma.from_documents(texts, embedding = embedding)
retriever = vectordb.as_retriever()

# Prompt
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Document is use only vietnamese language. So, you use it to reply.
Answer:
"""
prompt = PromptTemplate.from_template(template)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
     }
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in chain.stream('Tờ khai gì'):
    print(chunk)

# output = {}
# curr_key = None
# for chunk in rag_chain_with_source.stream("What is Task Decomposition"):
#     for key in chunk:
#         if key not in output:
#             output[key] = chunk[key]
#         else:
#             output[key] += chunk[key]
#         if key != curr_key:
#             print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
#         else:
#             print(chunk[key], end="", flush=True)
#         curr_key = key
# print(output)
