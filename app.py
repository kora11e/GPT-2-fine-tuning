from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm import load_llm

app = FastAPI()

llm = load_llm()

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer concisely: {question}"
)

chain = LLMChain(llm=llm, prompt=prompt)

class Query(BaseModel):
    question: str

@app.post("/generate")
def generate(query: Query):
    result = chain.run(query.question)
    return {"response": result}
