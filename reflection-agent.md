# Reflection Agent

[Reflection Agents](https://www.youtube.com/watch?v=v5ymBTXNqtk)에서는 Reflection Agent에 대해 설명하고 있습니다. 이와 관련된 [Blog - Reflection Agents](https://blog.langchain.dev/reflection-agents/)을 참조합니다. 

Reflection은 Agent을 포함한 AI 시스템의 품질과 성공률을 높이기 위해 사용되는 프롬프트 전략(prompting strategy)입니다. 

LangGraph를 사용하여 3가지 반영 기술을 구축하는 방법을 설명하고 있으며, Reflexion과 Language Agent Tree Search의 구현 방법도 포함되어 있습니다. 

## Simple Reflection

[agent-reflection-kor.ipynb](./agent/agent-reflection-kor.ipynb)에서는 Reflection을 구현하는 방법에 대해 설명합니다. 이때의 개념도는 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/2a77a177-5be9-4a7d-97a8-4d5a19f9709e)

*참고문헌*

- [agent-reflection.ipynb](./agent/agent-reflection.ipynb) 에서는 MessageGraph()로 LangGraph Agent 만드는것을 설명합니다.

- [reflection.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/reflection/reflection.ipynb)에서는 LangGraph로 Reflection을 이용한 Agent를 설명하고 있습니다. 이것은 re-planning, search, evalution에 활용될 수 있습니다. 




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



![image](https://github.com/user-attachments/assets/d40b049f-3fc3-4e26-909c-d04236b36c27)





## Reflexion

[reflexion.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/reflexion/reflexion.ipynb?ref=blog.langchain.dev)에서는 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)을 기반한 Reflexion을 구현하고 있습니다.

Reflexion의 Diagram은 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/469174cb-5ae9-444f-a19c-68261bab65dd)

feedback과 self-reflection을 이용해 더 높은 성능의 결과를 얻습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/fcaab550-b7ec-4edb-9fcf-576135075391)

Graph의 구현 코드는 아래와 같습니다. 

```python
MAX_ITERATIONS = 5
builder = MessageGraph()
builder.add_node("draft", first_responder.respond)

builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

builder.add_edge("draft", "execute_tools") # draft -> execute_tools
builder.add_edge("execute_tools", "revise") # execute_tools -> revise

def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i

def event_loop(state: list) -> Literal["execute_tools", "__end__"]:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop)  # revise -> execute_tools OR end
builder.set_entry_point("draft")
graph = builder.compile()
```

이것의 구현된 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/00f6d691-1b19-4fa9-9d1a-6049698d9d00)

## Language Agent Tree Search

[lats.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/lats/lats.ipynb?ref=blog.langchain.dev)에서는 reflection, evaluation, search을 이용해 전체적인 성능을 높입니다.

참고한 문헌은 [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/pdf/2310.04406)와 같습니다. 

Language Agent Tree Search (LATS)의 형태는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/09f9f7d1-bab2-4609-8ae5-dbe980b366fb)


이것읜 형태는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/92c34cf9-c3a2-4890-bd16-2856ebfde42a)

```python
builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.set_entry_point("start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
)

graph = builder.compile()
````

구현된 결과는 아래와 같습니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/bf61e626-638d-4e02-9835-5909822ae914)
