# 📝 Smart-Document-Summarization-and-QA-Assistantt

This repository contains a chatbot application developed using [Chainlit](https://www.chainlit.io/) and various AI-powered services. The chatbot is designed to summarize documents, and user-inputted text, with a focus on Vietnamese language documents.

## ✨ Features

- **Summarize Documents:** Upload PDF, Word, or text files, and the chatbot will summarize the content.
- **Summarize User Input:** Input any text, and the chatbot will provide a concise summary.
- **Vietnamese Language Support:** The chatbot is optimized for handling documents and content in Vietnamese.

## 📋 Prerequisites

- Python 3.9+
- Required libraries in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/document-summarization-chatbot.git
    cd document-summarization-chatbot
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
        GEMINI_API=<your_google_genai_api_key>
        LANGCHAIN_API_KEY=<your_langchain_api_key>
        ```

## 🚀 Usage

1. Start the chatbot:
    ```bash
    chainlit run app.py -w
    ```

2. Interact with the chatbot through the interface:
    - **Summarize Document:** Upload a PDF, Word, or text file for summarization.
    - **Summarize User Input:** Enter any text to receive a summary.
    - **Document QA:** Enter any question related to the document.

3. The chatbot will guide you through actions and provide summaries directly in the chat interface.

## 📁 Project Structure

- **`app.py`:** The main entry point for the chatbot application.
- **`Process_Document.py`:** Contains functions for document processing.
- **`document_summarize.py`:** Summarization functions used by the chatbot.
- **`Image/`:** Directory containing images used in the chatbot interface.
- **`Document/`:** Directory for storing sample documents.
- **`requirements.txt`:** List of required Python libraries.

## 🤖 Technologies Used

- **Chainlit:** Framework for building conversational AI applications.
- **LangChain:** Used for handling LLM (Large Language Model) operations.
- **Google Generative AI:** For text generation and embeddings.
- **FAISS** Vector databases for document retrieval and similarity search.
- **BeautifulSoup & PyPDFLoader:** For web scraping and PDF processing.

## 🎨 Customization

Feel free to modify the `template` in `app.py` to adjust the chatbot's prompt and response style according to your needs.

## 📈 Next Steps

- **User Question Suggestions:** Enhance the chatbot by suggesting potential questions that users can ask based on the content provided. This can improve user experience by guiding them to ask relevant and meaningful questions.

## 📧 Contact

For any queries or support, please contact [Lê Hữu Hưng](mailto:lehuuhung30023010@gmail.com).
