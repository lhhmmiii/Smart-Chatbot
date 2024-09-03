import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
google_api_key = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash', max_retries= 2, timeout= None, max_tokens = None, api_key = google_api_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_question_runnables(llm, retriever):

    template = """
    You are an expert in document analysis. I will provide you with a retriever that can access the content of the document. Your task is to generate a set of questions related to the document, using the same language as the document.

    Process:

    1. **Review the Document**: Use the retriever to extract key information, main topics, and important details from the document.
    2. **Generate Questions**: Based on the extracted content, create questions that cover the main ideas, specific details, potential implications, and any areas that may need further clarification or exploration.

    Make sure the questions are clear, concise, and varied in type (e.g., open-ended, factual, exploratory). The questions should be written in the only same language as the document.

    Document Language: {document_language}

    Document Content:
    {context}

    Questions:
    1. 
    2. 
    3. 
    ...
    """

    prompt = PromptTemplate.from_template(template)

    chain = (
        {  
            "document_language": RunnablePassthrough(),
            "context": retriever | format_docs
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# #############################################################

# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader

# load_dotenv()
# google_genai_api_key = os.getenv('GEMINI_API')



# # Load document
# loader = PyPDFLoader('7698_Cac-moc-thoi-gian-KLTN-TTTN-TTDATN-K2021-SV.pdf')
# docs = loader.load_and_split()

# # Split
# # Summarize
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
# splits = text_splitter.split_documents(docs)

# # Embedding
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
# vectostore = Chroma.from_documents(splits, embeddings)
# retriever = vectostore.as_retriever()
# #############################################################

# chain = generate_question_runnables(llm)

# print(chain.invoke("Vietnamese"))