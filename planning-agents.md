# Planning Agent
[LangGraph: Planning Agents](https://www.youtube.com/watch?v=uRya4zRrRx4)에서는 3가지 plan-and-execution 형태의 agent를 설명하고 있습니다. 

LangGraph은 staful하고 multi-actor 애플리케이션을 만들 수 있도록 돕는 오픈 소스 framework입니다. 이를 통해 빠르게 실행하고, 비용을 효율적으로 사용하고 성능을 향상 시킬 수 있습니다. 

## Basic Plan-and-Execute

[plan-and-execute.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb)에서는 [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)에 대한 Agent를 정의합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a97d0764-2891-4454-8854-522ef3249e44)

전체적인 구조는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3a311023-53d7-464a-b4a0-655c558bc058)

class와 함수를 정의합니다. 

```python
"system" = """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )    

async def plan_step(state: PlanExecute):  # planner

async def execute_step(state: PlanExecute):  # agent

async def replan_step(state: PlanExecute): # replan

def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
```

Graph, Node, Edge를 정의합니다.

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

실행은 아래와 같습니다.

```python
config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the 2024 Australia open winner?"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

## Reasoning without Observation

[rewoo.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb)에서는 multi-step planner를 진행할때 observation없이 사용하는 방법을 설명합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/ece962bf-d13a-459a-b547-23fc1dd018fc)

planner는 task 처리 형태는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3ff28ecd-67ff-4500-a8cb-8a7758de92be)

```python
class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str

def get_plan(state: ReWOO): # plan

def tool_execution(state: ReWOO): # tool

def solve(state: ReWOO):  # solve

def _get_current_task(state: ReWOO):

def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"
```

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

