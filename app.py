from flask import Flask, request, jsonify
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import configparser
from llm_tools import search_tool, wiki_tool, save_tool


config = configparser.ConfigParser()
config.read('config.ini')

OPENAI_API_KEY = config["OPENAI_API"]["API_KEY"]

app = Flask(__name__)

class ResearchTheResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]



@app.route("/api/v1/get_llm_response", methods=["POST"])
def get_query_response():

    query = request.get_json().get("query", "")
    if not query:
        return jsonify({"Error": "Query Required!!"}), 404
    
    llm = ChatOpenAI(model_name="gpt-4o", api_key=OPENAI_API_KEY)
    llm_parser = PydanticOutputParser(pydantic_object=ResearchTheResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use neccessary tools. 
                Wrap the output in this format and provide no other text\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=llm_parser.get_format_instructions())

    tools = [search_tool, wiki_tool, save_tool]
    
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    raw_response = agent_executor.invoke({"query": query})

    try:
        structured_response = llm_parser.parse(raw_response.get("output", ""))
        print("Structured Response: ", structured_response)
        return jsonify({"response": structured_response.model_dump()}), 200

    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

    
