import bs4
from langchain_community.document_loaders import WebBaseLoader

def extract_content_from_web(url):
    loader = WebBaseLoader(
        web_paths=([url]),
        # bs_kwargs=dict(
        #     parse_only=bs4.SoupStrainer(
        #         class_=("post-content", "post-title", "post-header")
        #     )
        # ),
    )
    docs = loader.load()
    return docs[0].page_content
