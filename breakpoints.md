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

workflow = StateGraph(State)
workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_node("step_3", step_3)
workflow.add_edge(START, "step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", "step_3")
workflow.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add 
graph = workflow.compile(checkpointer=memory, interrupt_before=["step_3"])
```

이를 통해 구현된 workflow는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/a52ab2ae-29a6-4a39-8b75-fb47fc166191)


이제 아래와 같이 입력을 지정하고 실행합니다. Step3에서 멈춘다음에 입력을 받으려고 대기 합니다. 

```python
initial_input = {"input": "hello world"}
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

user_approval = input("Do you want to go to Step 3? (yes/no): ")

if user_approval.lower() == 'yes':
    
    # If approved, continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
else:
    print("Operation cancelled by user.")
```

이때의 실행결과는 아래와 같습니다. Step 3을 실행하기 전에 멈춰선 후에 "yes"를 입력하면, breakpoint 이후로 실행을 계속합니다. 

```text
{'input': 'hello world'}
---Step 1---
---Step 2---
Do you want to go to Step 3? (yes/no):  yes
---Step 3---
```






