# 📝 Smart-Document-Summarization-and-QA-Assistantt

This repository contains a chatbot application developed using [Chainlit](https://www.chainlit.io/) and various AI-powered services. This chatbot can respond to generate and edit photos, can summarize text from users, and can answer document-related questions.

**Currently having a little problem because langchain's tools are only in beta.**

## ✨ Features

- **Summarize Content** Upload PDF, Word, or text files, and the chatbot will summarize the content. (The chatbot is optimized for handling documents and content in Vietnamese.)
- **Document QA:** The task of answering questions based on the content of a document using AI to understand and extract relevant information.
- **Generative and edit image:** The process of creating or modifying images using AI based on user-provided descriptions or instructions.
- **Question Answering:** Use current knowledge of the LLM(Gemini-1.5-flash) to answer the input from user.
- **Search:** A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query. **(processing)**

## Chatbot components
![Chatbot Components](https://github.com/lhhmmiii/Smart-Chatbot/blob/main/Image/Chatbot_components.png)

## 📋 Prerequisites

- Python 3.9+
- Required libraries in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/lhhmmiii/Smart-Chatbot.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add the following API keys:
        ```plaintext
        HUGGINGFACEHUB_API_TOKEN = <your_huggingface_api_token>
        GEMINI_API = <your_gemini_api>
        TAVILY_API_KEY = <your_tavily_api_key>
        LITERAL_API_KEY = <your_literal_api_key>
        CHAINLIT_AUTH_SECRET = <your_chainlit_auth_api_key>
        STABILITY_KEY = <your_stability_api_key>
        ```

## 🚀 Usage

1. Start the chatbot:
    ```bash
    chainlit run app.py -w
    ```
2. Log in

- **Acount:** LHH
- **Password:** 1323

Log in to save your chat history, ensuring you can pick up right where you left off at any time."

3. Interact with the chatbot through the interface:
    - **User Input:** If you want to summarize text then put text in and add command to summarize text, same with generating and editing images. (I use agent to do this task)
    - **Document QA:** Enter any question related to the document.

4. The chatbot will guide you through actions and provide summaries directly in the chat interface.

## 📁 Project Structure

- **`app.py`:** The main entry point for the chatbot application.
- **`Process_Document.py`:** Contains functions for document processing.
- **`document_summarize.py`:** Summarization functions used by the chatbot.
- **`Image/`:** Directory containing images used in the chatbot interface and generated image.
- **`tools.py`:** contain tools which agent choose
- **`history_chatbot.py`:** history of chatbot
- **`requirements.txt`:** List of required Python libraries.

## 🤖 Technologies Used

- **Chainlit:** Framework for building conversational AI applications.
- **LangChain:** Used for handling LLM (Large Language Model) operations.
- **FAISS** Vector databases for document retrieval and similarity search.
- **PyPDFLoader:** For web scraping and PDF processing.


## 🎨 Customization

Feel free to modify the `template` in `app.py` to adjust the chatbot's prompt and response style according to your needs.

## 📈 Next Steps

- Done search tools

## 📧 Contact

For any queries or support, please contact [Lê Hữu Hưng](mailto:lehuuhung30023010@gmail.com).
