# Breakpoints

[breakpoints.ipynb](./agent/breakpoints.ipynb)에서는 breakpoint의 개념과 사용예를 보여줍니다. 이 노트북의 원본은 [langchain-breakpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)입니다. 

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    pass

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()
