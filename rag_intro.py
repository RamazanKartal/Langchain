from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


loader = WebBaseLoader(
    #web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),       #use this if you want to take more than 1 websites
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")    # this can be different for every website
        )
    )   
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


if __name__ == "__main__":
    user_input = input("What is your question: ")
    
    #for chunk in rag_chain.stream("What is MIPS?") :
    for chunk in rag_chain.stream(user_input):
        print(chunk, end="", flush=True)
  
    