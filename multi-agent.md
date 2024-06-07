# Multi Agent

[LangGraph: Multi-Agent Workflows](https://www.youtube.com/watch?v=hvAPnpSfSGo&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&index=10)에서 설명하고 있는 3가지 multi agent에 대해 정리합니다. 

## Basic Multi-agent Collaboration

[multi-agent-collaboration.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/multi-agent-collaboration.ipynb)에서는 여러 agent들이 서로 협력하는 방법을 설명하고 있습니다. 

[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/pdf/2308.08155)을 참조하였습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/518a970a-87d8-4637-a152-f3fab96e2984)

이때의 구조는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/01ddaae6-a656-41d6-afc5-f60d4d672c32)

구현 코드는 아래와 같습니다.

```python
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()
```

## Agent Supervisor

다른 여러개의 Agent를 orchestration하는 방법에 대해 설명합니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/746af98d-1cee-4659-9783-f17d0eb0c4b1)

```python
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

graph = workflow.compile()
```

