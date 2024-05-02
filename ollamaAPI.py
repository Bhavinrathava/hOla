import ollama
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from starlette.responses import Response

from typing_extensions import TypedDict
from typing import List

from pinecone import Pinecone

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
import os

import ollama



local_embedding = 'mxbai-embed-large'
local_llm = 'llama3'

ollama.pull(model=local_embedding)
ollama.pull(model = local_llm)
app = FastAPI()


def format_docs(docs):
    formatted_docs = ""
    for doc in docs:
        formatted_docs += doc + " "
    return formatted_docs

@app.get("/")
def root():
    return {"message": "Fast API in Python"}


@app.get("/getLlama/{prompt}")
def getLlama():
    pass


@app.get("/embed/", status_code=200)
def get_embedding(prompt: str, response: Response):
    return ollama.embeddings(model="mxbai-embed-large", prompt=prompt)['embedding']
    
# TODO : 1. Change the method to PUT
# TODO : 2. Change the prompt to generate answers based on PDF and question
# TODO : 3. Refer the Original LLM_Extraction for the prompt 
@app.get("/generate/", status_code=200)
def generate(question: str, documents: str, response: Response):
        # Generate the answer based on the question and the documents 
        prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
        )

        llm = ChatOllama(model=local_llm, temperature=0)
        
        print("Initialising the chain...")
        rag_chain = prompt | llm | StrOutputParser()

        # Generate the answer
        return rag_chain.invoke({"question": question, "context": (documents)})


@app.get("/route/", status_code=200)
def get_route(question: str, response: Response):
    print("Routing the question to the appropriate datasource...")
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore , web search, or Canvas (student classroom schedule and course management portal). 
        Use the vectorstore for questions on AI, Machine Learning, DataScience and other relevant CS related questions.You do not need to be stringent with the keywords in the question related to these topics. 
        Use Canvas to get information about student's class schedule, grades and other school related information. 
        Use the web search for all other questions.
        Give a choice between 'web_search' or 'vectorstore' or 'canvas_search' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    datasource = result['datasource']

    if(datasource == "web_search"):
        return "web_search"
    elif(datasource == "vectorstore"):
        return "retrieve"
    else:
        return "canvas_search"


if __name__ == "__main__":
    uvicorn.run(app, port=int(os.environ.get("PORT", 8081)))