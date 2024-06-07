# Reflection Agent


[reflection.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/reflection/reflection.ipynb)에서는 LangGraph로 Reflection을 이용한 Agent를 설명하고 있습니다. 이것은 re-planning, search, evalution에 활용될 수 있습니다. 


![image](https://github.com/kyopark2014/llm-agent/assets/52392004/7ceb3d72-7fc3-4939-bdd0-f0260121e498)

이것을 구현한 코드는 아래와 같습니다.

```python
builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        # End after 3 iterations
        return END
    return "reflect"

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()
```
