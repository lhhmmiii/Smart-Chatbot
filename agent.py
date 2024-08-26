# from langchain import hub
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.tools.retriever import create_retriever_tool
# from langchain.agents import create_tool_calling_agent
# from langchain.agents import AgentExecutor

# # Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt.messages)

# def my_tools(retriever):
#     search = TavilySearchResults(max_results=2)
#     retriever_tool = create_retriever_tool(
#         retriever,
#         "rag_retriever", # name
#         "Searches information from my documents", # description
#     )
#     tools = [retriever_tool, search]

#     return tools


# def create_agent_executor(llm, retriever, prompt = prompt):
#     tools = my_tools(retriever)
#     agent = create_tool_calling_agent(llm, tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools)
#     return agent_executor

