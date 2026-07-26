"""
This module contains the ExploreAndExecuteAgent class, which is used to explore and execute the agent.
"""
import json
from jinja2 import Template
import warnings
import unittest
from typing import Any, Dict
from llm.base import BaseLLM
from mcplib.client import MCPClient
from llm.gpt import OpenAILLM
import asyncio


REACT_PROMPT = """
You are a ReAct Agent assistant help user to answer questions about News. You will be given a user question and images (optional). You need to complete the task with provided tools.

## User Question
<user_question>
{{ user_question }}

## News Title or Text
<news_text>
{{ news_text }}
</news_text>

{% if news_image %}
## News Image
Image Path: {{ news_image }}

<image_placeholder>
{% endif %}

{% if available_tools %}
<available_tools>
## Available Tools
{{ available_tools }}
</available_tools>
{% endif %}

## Execution History
<history>
{{ history }}
</history>

## Response
- if you have completed the task, please output the final report or answer to the user question.
return in the following format:
{
  "answer": <final answer to the user question, string>,
  "reason": <reason of the answer>
}

{% if available_tools %}
- if you need to extra information, please call a tool, and output the tool name and arguments. 
return in the following format:
{
  "server_name": <server name>,
  "tool_name": <tool name>,
  "tool_args": <tool arguments>,
  "thought": <thoughts of the tool call, string>
}
{% endif %}

# Important: Only output the final answer with json format. No other text.

Your answer:
"""

class ReActAgent():
    """
    A class that explores and executes the agent.
    """
    def __init__(self, mcp_client: MCPClient, llm: BaseLLM, max_iterations: int=25):
        super().__init__()
        self.llm = llm
        self.history = []
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    def _execute(self, question: str, context: Dict[str, Any] = {}):
        """
        Execute the agent.
        """ 
        template = Template(REACT_PROMPT)
        execute_prompt = template.render(
            user_question=question,
            history=self.history,
            **context
        )
        execute_response = self.llm.generate_response(execute_prompt)
        # replace the ```json at the beginning of string
        if isinstance(execute_response, str):
            execute_response = execute_response.replace("```json", "").replace("```", "")
            print("Execute Response: \n", execute_response)
            execute_response_json = json.loads(execute_response)
        else:
            execute_response_json = execute_response
        print("Execute Response: \n", execute_response_json)
        return execute_response_json

    async def execute(self, question: str, doc: str = None, context: Dict[str, Any] = {}):
        """
        Execute the agent.
        """

        for iteration in range(self.max_iterations):
            print("="*100)
            print(f"Iteration {iteration+1}")
            print("="*100)
            # execute the execute step
            execute_response_json = self._execute(question, context)

            # if the plan is completed, return the answer
            if "answer" in execute_response_json:
                print("Final Answer: \n", execute_response_json["answer"])
                yield execute_response_json
                return

            # if need to call a tool, call the tool
            elif "tool_name" in execute_response_json:
                server_name = execute_response_json["server_name"]
                tool_name = execute_response_json["tool_name"]
                tool_args = execute_response_json["tool_args"]
                self.history.append(f"Tool Call Parameters:\n {tool_name}\n {tool_args}")
                yield execute_response_json
                await asyncio.sleep(0.1)

                tool_response = await self.mcp_client.tool_call(server_name, tool_name, tool_args)
                self.history.append(f"Tool Call Result:\n {tool_response}")

                yield {"mcp_response": tool_response}
                await asyncio.sleep(0.1)
            else:
                warnings.warn(f"No tool call found in the execute response: {execute_response_json}")
