from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.1)

messages = [
    SystemMessage(content="Translate the following text from English to German"),
    HumanMessage(content="What is Langchain?")
]

parser = StrOutputParser()

chain = model | parser

if __name__ == "__main__":
    
    #print(response.content)
    print(chain.invoke(messages))
    

