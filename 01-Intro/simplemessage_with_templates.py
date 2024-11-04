from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.1)

system_prompt = "Translate the following text into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), ("user", "{text}")
    ]
)

parser = StrOutputParser()
chain = prompt_template | model | parser


if __name__ == "__main__":
    
    print(f"Translates the given text into desired language")
    text = input("Enter the text for translation:")
    language = input("To which language:")
    
    print(chain.invoke({"language": language, "text": text}))
    #print(chain.invoke({"language": "Italian", "text": "Hello World!"}))

