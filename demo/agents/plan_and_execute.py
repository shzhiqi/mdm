"""
This module contains the ExploreAndExecuteAgent class, which is used to explore and execute the agent.
"""
import json
from tkinter import N
from jinja2 import Template
import warnings
import unittest
from typing import Any, Dict
from llm.base import BaseLLM
from mcplib.client import MCPClient
from llm.gpt import OpenAILLM
import builtins

PLAN_PROMPT = """
You are an AI assistant help user to answer questions about News. You will be given a user question, and you need to plan the next steps to answer the user question.

## User Question
<user_question>
{{ user_question }}
</user_question>

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
## Available Tools Description
<available_tools>
{{ available_tools }}
</available_tools>
{% endif %}

Please directly output the plan steps and status in markdown check list format:
- [ ] Step 1
- [ ] Step 2
- [ ] Step 3
- more steps...

Important:
- If user questions is explicitly need the modify the final report, response the modified report. otherwise, direcltly answer the user question.
- Only call the tool if you have to.
- Each step is either a plan step that can be directly answered or a tool call.

Your Response:
"""

EXECUTE_PROMPT = """
You are an AI assistant help user to answer questions about News. You will be given a user question, a plan and status. You need to execute the plan step by step based on the plan and status.

If you have completed one step of the plan, please update the plan and status.

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

## Plan and Status
<plan_and_status>   
{{ plan_and_status }}
</plan_and_status>

## Response
- if you have completed some steps of the plan, please output the final report or answer to the user question.
return in the following format:
{
  "answer": <final report or answer to the user question>,
  "reason": <reason of the answer>,
  "plan_and_status": <updated plan and status, markdown checklist format>
}

{% if available_tools %}
- if you need to call a tool, please output the tool name and arguments. 
return in the following format:
{
  "tool_name": <tool name>,
  "tool_args": <tool arguments>,
  "reason": <reason of the tool call>,
  "plan_and_status": <updated plan and status, markdown checklist format>
}
{% endif %}

- if you have not completed the plan, but completed some steps of the plan, please mark the completed steps in the plan and status.
return in the following format:
{
  "output": <next step of the plan>,
  "reason": <reason of the output, whether you have completed this step>,
  "plan_and_status": <updated plan and status, markdown checklist format>
}
"""
# overwrite print function, also write to log file  
def print(*args, **kwargs):
    with open("log.txt", "a", encoding="utf-8") as f:
        # without color 
        for text in args:
            if not isinstance(text, str):
                text = str(text)
            clean_text = text.replace("\033[92m", "").replace("\033[91m", "").replace("\033[93m", "").replace("\033[0m", "")
            f.write(clean_text+"\n") 
    builtins.print(*args, **kwargs)

class PlanAndExecuteAgent():
    """
    A class that plans and executes the agent.
    """
    def __init__(self, mcp_client: MCPClient, llm: BaseLLM, max_iterations: int=25):
        super().__init__()
        self.llm = llm
        self.history = []
        self.plan_and_status: str = ""
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    def _plan(self, question: str, context: Dict[str, Any] = {}):
        """
        Plan the next steps to complete the task.
        required context:
        - final_report: the final report
        - intermediate_reports: the intermediate reports
        - available_tools: the available tools
        """
        template = Template(PLAN_PROMPT)
        plan_prompt = template.render(
            user_question=question,
            **context
        )
        execute_response = self.llm.generate_response(plan_prompt, images=context.get("news_image", None))
        print("Plan Response: \n", execute_response)
        return execute_response

    def _execute(self, question: str, context: Dict[str, Any] = {}):
        """
        Execute the plan.
        required context:
        - final_report: the final report
        - intermediate_reports: the intermediate reports
        - available_tools: the available tools
        """
        template = Template(EXECUTE_PROMPT)
        execute_prompt = template.render(
            user_question=question,
            history=self.history,
            plan_and_status=self.plan_and_status,
            **context
        )
        execute_response = self.llm.generate_response(execute_prompt, images=context.get("news_image", None))
        # replace the ```json at the beginning of string
        execute_response = execute_response.replace("```json", "").replace("```", "")
        execute_response_json = json.loads(execute_response)
        print("Execute Response: \n", execute_response_json)
        return execute_response_json

    async def execute(self, question: str, doc: str = None, context: Dict[str, Any] = {}):
        """
        Execute the agent.
        """
        # execute the plan step
        self.plan_and_status = self._plan(question, context)
        self.history.append(f"Plans to complete the task: {self.plan_and_status}")

        for iteration in range(self.max_iterations):
            print("-"*66)
            print(f"Iteration: {iteration}")

            # execute the execute step
            execute_response_json = self._execute(question, context)

            # if the plan is completed, return the answer
            is_completed = False
            if "answer" in execute_response_json:
                self.plan_and_status = execute_response_json["plan_and_status"]
                print("Final Answer: \n", execute_response_json["answer"])
                is_completed = True

            # if need to call a tool, call the tool
            elif "tool_name" in execute_response_json:
                tool_name = execute_response_json["tool_name"]
                tool_args = execute_response_json["tool_args"]
                self.history.append(f"Tool Call Parameters:\n {tool_name}\n {tool_args}")
                tool_response = await self.mcp_client.tool_call(tool_name, tool_args)
                self.history.append(f"Tool Call Result:\n {tool_response}")
                self.plan_and_status = execute_response_json["plan_and_status"]

                # yellow color for the tool call
                print("\n"+"-"*66+"\n")
                print(f"\033[93mTool Call: \n{tool_name}\n {tool_args}\033[0m")
                print(f"\033[93mTool Call Result: \n{tool_response}\033[0m")

            elif "output" in execute_response_json:
                self.plan_and_status = execute_response_json["plan_and_status"]
                self.history.append(f"Current Output:\n {execute_response_json['output']}")

            else:
                warnings.warn(f"No plan or tool call found in the execute response: {execute_response_json}")
            
            # green color for the plan and status
            print("-"*66)
            print(f"\033[92mPlan and Status: \n{self.plan_and_status}\033[0m")

            if is_completed:
                # red color for the final answer
                print("-"*66)
                print(f"\033[91mFinal Answer: \n{execute_response_json['answer']}\033[0m")
                print(f"\033[91mReasons of the answer: \n{execute_response_json['reason']}\033[0m")
                print("-"*66)
                return execute_response_json["answer"]