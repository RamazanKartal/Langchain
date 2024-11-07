from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


#  Collections are the tables in SQL, rows(instances) in a table are the documents in vectordb
embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# we ask from retriever to run the similarity_search()
retreiver = RunnableLambda(vectorstore.similarity_search).bind(k=1)
model = ChatOpenAI(model="gpt-4o-mini")

message = """
Answer the question only with the provided context.

{question}

context: {context}

"""

prompt = ChatPromptTemplate.from_messages(
    [("human", message)]
)

#RunnablePassthrough is a place holder for us, we will give the input later
chain = {"context": retreiver, "question": RunnablePassthrough()} | prompt | model


if __name__ == "__main__":
    #print(vectorstore.similarity_search("dog"))
    #embedding = OpenAIEmbeddings().embed_query("dog")
    #print(vectorstore.similarity_search_by_vector(embedding))
    #print(vectorstore.similarity_search_with_score("dog"))
    
    response = chain.invoke("tell me about birds")
    print(response.content)
    
    
