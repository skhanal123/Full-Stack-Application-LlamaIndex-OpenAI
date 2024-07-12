from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import numpy as np
import os.path
import os

import nest_asyncio

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "*****"

nest_asyncio.apply()

app = FastAPI()

origins = ["http://127.0.0.1:5500", "*", "http://127.0.0.1:5000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_DIR = "./storage"


try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./researchPaper/critical_care_nepal"
    )
    critical_care_nepal_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./researchPaper/covid_19_pathophysiology"
    )
    covid_19_pathophysiology_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

critical_care_engine = critical_care_nepal_index.as_query_engine(similarity_top_k=3)
covid_19_engine = covid_19_pathophysiology_index.as_query_engine(similarity_top_k=3)


query_engine_tools = [
    QueryEngineTool(
        query_engine=critical_care_engine,
        metadata=ToolMetadata(
            name="critical_care_nutrition",
            description=(
                "Provides information about status of critical care nutrition in Nepal. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=covid_19_engine,
        metadata=ToolMetadata(
            name="covid_19_pathophysiology",
            description=(
                "Provides information about covid 19 pathophysiology. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]


context = """\
You are a health care expert who has good knowledge about status of critical care nutrition in Nepal and Covid 19 pathophysiology.\
    You will answer questions about critical care nutrition and covid-19 pathophysiology as in the persona of a health care expert.
"""
llm = OpenAI(model="gpt-3.5-turbo-0613")

agent = ReActAgent.from_tools(
    query_engine_tools, llm=llm, verbose=True, context=context
)


@app.get("/ping")
async def checkServer():
    return "The server is working fine"


@app.post("/userQuery")
async def queryResponse(request: Request):
    query1 = await request.body()
    decoded_data = query1.decode("utf-8")
    parsed_json = json.loads(decoded_data)

    print(parsed_json["query"])

    response1 = agent.chat(parsed_json["query"])
    print(str(response1))
    return {"response": response1}


if __name__ == "__main__":
    uvicorn.run(app, port=5000, host="127.0.0.1")
