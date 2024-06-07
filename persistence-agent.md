# Persistence Agent

[persistence.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/persistence.ipynb)에서는 checkpoint를 이용해 state를 관리하는것을 보여줍니다.

```python
workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("action", "agent")
```

이때의 Graph는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4600a709-ec26-4684-88ed-2060b3b41813)

## Interacting with the Agent

메모리를 이용합니다.

```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```
