# Planning Agent
[LangGraph: Planning Agents](https://www.youtube.com/watch?v=uRya4zRrRx4)에서는 3가지 plan-and-execution 형태의 agent를 설명하고 있습니다. 

LangGraph은 staful하고 multi-actor 애플리케이션을 만들 수 있도록 돕는 오픈 소스 framework입니다. 이를 통해 빠르게 실행하고, 비용을 효율적으로 사용하고 성능을 향상 시킬 수 있습니다. 

## Basic Plan-and-Execute

[/plan-and-execute.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb)에서는 [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)에 대한 Agent를 정의합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a97d0764-2891-4454-8854-522ef3249e44)

전체적인 구조는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3a311023-53d7-464a-b4a0-655c558bc058)

```python
from langgraph.graph import StateGraph

workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
)

app = workflow.compile()
```


## Reasoning without Observation

[rewoo.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb)에서는 multi-step planner를 진행할때 observation없이 사용하는 방법을 설명합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/ece962bf-d13a-459a-b547-23fc1dd018fc)

planner는 task 처리 형태는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3ff28ecd-67ff-4500-a8cb-8a7758de92be)

이때의 Graph 구성은 아래와 같습니다. 

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

app = graph.compile()
```

## LLMCompiler

[LLMCompiler.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb)에서는 "An LLM Compiler for Parallel Function Calling"을 구현한 것을 설명하고 있습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c17e641b-93eb-451d-9020-be198ae184fc)

Task fetching unit

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4daeafb1-b804-441c-91d5-dad30558c261)


```python
from langgraph.graph import MessageGraph, END
from typing import Dict

graph_builder = MessageGraph()

graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge("plan_and_schedule", "join")

def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"

graph_builder.add_conditional_edges(
    start_key="join",
    condition=should_continue,
)

graph_builder.set_entry_point("plan_and_schedule")

chain = graph_builder.compile()
```

