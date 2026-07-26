from re import S
import gradio as gr
import asyncio
from agents.react import ReActAgent
from llm.gpt import OpenAILLM
from llm.claude import ClaudeLLM
from llm.gemini import GeminiLLM
from mcplib.client import MCPClient
from pathlib import Path
import os
import json

custom_css = """
    hr { margin-bottom: 5px; margin-top: 5px; }
"""

async def misinformation_fn(title, img, user_question, img_cap_algo, llm_model, mcp_servers):   
    try:
        print(title, img) 

        yield gr.update(value="Connecting to MCP server..."), gr.update()

        # mcp_client = MCPClient(server_script_path="mcp_server/server.py")
        mcp_client = MCPClient(server_script_paths=[
            {
                "name": "HR-MCP",
                "server_script_path": "mcp_server/server.py"
            },
            # {
            #     "name": "Yahoo Finance",
            #     "server_script_path": "/Users/zhiqi.shen/Downloads/yahoo-finance-mcp/server.py"
            # }
        ])
        await mcp_client.connect_to_server()
        available_tools = await mcp_client.get_available_tools()
        print(available_tools)

        yield gr.update(value="MCP server connected"), gr.update()
        yield gr.update(value="Getting available tools..."), gr.update()

        available_tools_str = "<div style='font-size: 1.3em; font-weight: bold;'>MCP Available Tools:</div>"
        available_tools_str += "<hr>"
        for tool in available_tools:
            available_tools_str += f"<span><b>Tool Name:</b> {tool['name']}</span><br>"
            available_tools_str += f"<span><b>Tool Description:</b> {tool['description']}</span><br>"
            available_tools_str += "<hr>"
        yield gr.update(value=available_tools_str), gr.update()

        img.save("tmp.png")
        context = {
            "news_text": title,
            "news_image":[
                str(Path(os.path.join(os.path.dirname(__file__), "tmp.png")).resolve())
            ],
            "available_tools": available_tools
        }

        # # agent settings and execute
        llm = None
        if llm_model == "ChatGPT-4o":
            llm = OpenAILLM(model="gpt-4o")
        elif llm_model == "Claude-3.7-sonnet":
            llm = ClaudeLLM(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        elif llm_model == "Gemini-2.5-flash":
            llm = GeminiLLM(model="gemini-2.5-flash")
        max_iterations = 25
        agent = ReActAgent(mcp_client, llm, max_iterations)

        async_gen = agent.execute(user_question, context=context).__aiter__()

        try:
            iter_count = 1
            return_text = ""
            while True:
                iter_answer = await async_gen.__anext__()

                print(f"iter_count: {iter_count}, iter_answer: {iter_answer}")

                if "tool_name" in iter_answer:
                    iter_text = f"<div style='font-size: 1.3em; font-weight: bold;'>ReAct Step: {iter_count}</div>"
                    iter_text += f"<span><b><u>Thought:</u></b> {iter_answer['thought']}</span><br>"
                    iter_text += f"<span><b><u>Action:</u></b> Call {iter_answer['tool_name']} with Parameters: {iter_answer['tool_args']}</span><br>"
                    iter_text = iter_text.replace("/Users/zhiqi.shen/Code/MDM/code/demo/", "")
                    for text in str(iter_text).split(" "): 
                        return_text += text + " "
                        yield gr.update(), gr.update(value=return_text)
                        await asyncio.sleep(0.1)
                    
                if "mcp_response" in iter_answer:
                    iter_text = f"<span><b><u>Observation:</u></b> (Responses from <b>MCP server</b>)</span><br>"
                    responses = json.loads(iter_answer['mcp_response'])

                    iter_text += str(responses)

                    # for text_idx, (key, value) in enumerate(responses.items(), start=1):
                    #     iter_text += f"<span><b>{text_idx}. {key}:</b> {float(value):.3f}</span><br>"
                    #     if text_idx == 1:
                    #         yield gr.update(), gr.update(value=iter_text)
                    #         await asyncio.sleep(0.1)

                    for text in str(iter_text).split(" "): 
                        return_text += text + " "
                        yield gr.update(), gr.update(value=return_text)
                        await asyncio.sleep(0.1)
                    iter_count += 1

                elif "answer" in iter_answer:
                    iter_text = f"<div style='font-size: 1.3em; font-weight: bold;'>ReAct Step: {iter_count}</div>"
                    iter_text += f"<span><b><u>Final Answer:</u></b> {iter_answer['answer']}</span><br>"
                    for text in str(iter_text).split(" "): 
                        return_text += text + " "
                        yield gr.update(), gr.update(value=return_text)
                        await asyncio.sleep(0.1)
                    iter_count += 1
                
        except StopAsyncIteration:
            pass

    finally:
        await mcp_client.cleanup()

interface = gr.Interface(
    fn=misinformation_fn,
    inputs=[
        gr.Textbox(
            label="News Headline",
            lines=1),
        gr.Image(
            label='News Image',
            image_mode="RGB",
            type='pil'),
        gr.Textbox(
            label="User Question",
            lines=1),
        gr.Dropdown(
            ["ReAct", "Plan-and-Execute"],
            value="ReAct",
            label="Agent"),
        gr.Dropdown(
            ["ChatGPT-4o", "Claude-3.7-sonnet", "Gemini-2.5-flash"],
            value="ChatGPT-4o",
            label="LLM"),
        gr.CheckboxGroup(
            choices=["HR-MCP", "Yahoo Finance"],
            label="Choose plug-and-play MCP Servers",
            # value=[""]  # default selected
        )
    ],
    outputs=[
        gr.HTML(),
        gr.HTML()
    ],
    examples=[
        # ['Joel Osteen and His Wife, Victoria Osteen, Announced in June 2023 That They Would Be Resigning from Their Pastorship with Lakewood Church','./1006_snpimage.jpg', "Please evaluate human reactions to this post: Do they believe it's true? Will they share it? Could it go viral?", 'ReAct','ChatGPT-4o'],
        ['Joel Osteen and His Wife, Victoria Osteen, Announced in June 2023 That They Would Be Resigning from Their Pastorship with Lakewood Church','./1006_snpimage.jpg', "", 'ReAct','ChatGPT-4o'],
        ['San Francisco Mother and Child Forced to Live in Car Due to Skyrocketing Housing Costs','./demo2.jpg', "", '',''],
        ['Shocking Scene Unveiled: Rogue Cop Flaunts Authority, Lights Up Cigarette Outside Caracas Bar!', './demo3.jpg', "", '',''],
    ],
    title="T-Lens demo",
    description="Answer questions with human responses",
    flagging_mode="never",
    theme="light",
    css=custom_css
)
interface.launch(share=False, server_name="0.0.0.0")