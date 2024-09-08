import json

from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

# Tool definition
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)  # Its the actual tool from langchain for Tavily
tool_executor = ToolExecutor([tavily_tool])  # This is required as we want to run multiple tools in async


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]  # state[-1] means length-1
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query
            ))
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations)
    output_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        output_map[id_][invocation.tool_input] = output

    # Convert the mapped outputs to ToolMessage objects
    tool_messages = []
    for id_, mapped_output in output_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_)
        )

    return tool_messages


if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
                " list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_KpYHichFFEmLitHFvFhKy1Ra",
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    }
                ],
            ),
        ]
    )
    print(raw_res)
