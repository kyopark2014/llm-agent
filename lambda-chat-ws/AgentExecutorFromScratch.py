# Agent Executor From Scratch
import operator

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

def get_react_prompt_template(mode: str): # (hwchase17/react) https://smith.langchain.com/hub/hwchase17/react
    # Get the react prompt template
    
    if mode=='eng':
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should use only the tool name from [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
    else: 
        return PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

다음의 format을 사용하세요.:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action로서 [{tool_names}]에서 tool의 name만을 가져옵니다. 
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 5번 반복 할 수 있습니다.)
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

너는 Human에게 해줄 응답이 있거나, Tool을 사용하지 않아도 되는 경우에, 다음 format을 사용하세요.:
'''
Thought: Tool을 사용해야 하나요? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

def get_react_chat_prompt_template(mode: str):
    # Get the react prompt template
    if mode=='eng':
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should use only the tool name from [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Thought:{agent_scratchpad}
""")
    else: 

        return PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

다음의 format을 사용하세요.:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action로서 [{tool_names}]에서 tool의 name만을 가져옵니다. 
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 5번 반복 할 수 있습니다.)
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

너는 Human에게 해줄 응답이 있거나, Tool을 사용하지 않아도 되는 경우에, 다음 format을 사용하세요.:
'''
Thought: Tool을 사용해야 하나요? No
Final Answer: [your response here]
'''

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Thought:{agent_scratchpad}
""")
        
agentLangMode = 'kor'        

tools = []
class AgentState(TypedDict):
        input: str
        chat_history: list[BaseMessage]
        agent_outcome: Union[AgentAction, AgentFinish, None]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        
class AgentExecuteFromScratch:
    chat: any
    
    def __init__(self, tools):
        #self.chat = chat
        self.tools = tools
    
    def buildAgent():
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        return workflow.compile()    

    def buildAgentChat():
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", run_agent_chat)
        workflow.add_node("action", execute_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        return workflow.compile()    

    app_chat = buildAgentChat()

def run_agent(state: AgentState):
        agent_outcome = agent_runnable.invoke(state)
        return {"agent_outcome": agent_outcome}
    
def run_agent_chat(data):
        agent_outcome = agent_chat_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

def execute_tools(state: AgentState):
        agent_action = state["agent_outcome"]
        #response = input(prompt=f"[y/n] continue with: {agent_action}?")
        #if response == "n":
        #    raise ValueError
        
        tool_executor = ToolExecutor(tools)
        
        output = tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(state: AgentState):
        if isinstance(state["agent_outcome"], AgentFinish):
            return "end"
        else:
            return "continue"
            
prompt_template = get_react_prompt_template(agentLangMode)
agent_runnable = create_react_agent(chat, tools, prompt_template)
# LangGraph Agent (Chat)
prompt_chat_template = get_react_chat_prompt_template(agentLangMode)
agent_chat_runnable = create_react_agent(chat, tools, prompt_chat_template)
app = buildAgent(agent.run_agent, execute_tools, should_continue)