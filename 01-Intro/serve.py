from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langserve import add_routes
from fastapi import FastAPI


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

app = FastAPI(
    title="Translater App",
    version = "1.0.0",
    description="Translation Chat Bot"
)

add_routes(
    app,
    chain,
    path = "/chain" # www.google.com/chain
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
    

