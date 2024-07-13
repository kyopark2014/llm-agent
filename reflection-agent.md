# Reflection Agent

[Reflection Agents](https://www.youtube.com/watch?v=v5ymBTXNqtk)에서는 Reflection Agent에 대해 설명하고 있습니다. 이와 관련된 [Blog - Reflection Agents](https://blog.langchain.dev/reflection-agents/)을 참조합니다. 

Reflection은 Agent을 포함한 AI 시스템의 품질과 성공률을 높이기 위해 사용되는 프롬프트 전략(prompting strategy)입니다. 

LangGraph를 사용하여 3가지 반영 기술을 구축하는 방법을 설명하고 있으며, Reflexion과 Language Agent Tree Search의 구현 방법도 포함되어 있습니다. 

## Simple Reflection

[agent-reflection-kor.ipynb](./agent/agent-reflection-kor.ipynb)에서는 Reflection을 구현하는 방법에 대해 설명합니다. 이때의 개념도는 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/2a77a177-5be9-4a7d-97a8-4d5a19f9709e)

### 참고문헌

- [agent-reflection.ipynb](./agent/agent-reflection.ipynb) 에서는 MessageGraph()로 LangGraph Agent 만드는것을 설명합니다.

- [reflection.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/reflection/reflection.ipynb)에서는 LangGraph로 Reflection을 이용한 Agent를 설명하고 있습니다. 이것은 re-planning, search, evalution에 활용될 수 있습니다. 

### Node의 정의

에세이 형태의 Prompt를 구성합니다. 

```python
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 5문단의 에세이 작성을 돕는 작가이고 이름은 서연입니다"
            "사용자의 요청에 대해 최고의 에세이를 작성하세요."
            "사용자가 에세이에 대해 평가를 하면, 이전 에세이를 수정하여 답변하세요."
            "완성된 에세이는 <result> tag를 붙여주세요.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | chat
```

Reflect를 위한 Prompt를 정의합니다.

```python
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 교사로서 학셍의 에세이를 평가하삽니다. 비평과 개선사항을 친절하게 설명해주세요."
            "이때 장점, 단점, 길이, 깊이, 스타일등에 대해 충분한 정보를 제공합니다."
            "특히 주제에 맞는 적절한 예제가 잘 반영되어있는지 확인합니다",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | chat
```

Workflow를 위한 Node 함수를 정의합니다.

```python
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, List, Union

class ChatAgentState(TypedDict):
    # messages: Annotated[Sequence[BaseMessage], operator.add]
    messages: Annotated[list, add_messages]

def generation_node(state: ChatAgentState):    
    response = chain.invoke(state["messages"])
    return {"messages": [response]}

def reflection_node(state: ChatAgentState):
    messages = state["messages"]
    
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = reflect.invoke({"messages": translated})    
    response = HumanMessage(content=res.content)    
    return {"messages": [response]}

def should_continue(state: ChatAgentState):
    messages = state["messages"]
    
    if len(messages) >= 6:   # End after 3 iterations        
        return "end"
    else:
        return "continue"
```

StateGraph를 이용해 workflow를 정의합니다.

```python
from langgraph.graph import START, END, StateGraph

workflow = StateGraph(ChatAgentState)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)
workflow.set_entry_point("generate")
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "continue": "reflect",
        "end": END,
    },
)

workflow.add_edge("reflect", "generate")
app_reflection = workflow.compile()
```

구현된 workflow는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/b2cccf4d-8a91-4955-9e32-330f77182cff)

이제 아래와 같이 실행합니다.

```python
query = "한국 인공지능 발전을 어떤 준비를 해야할지 설명하세요. 특히 한국의 현황과 향후 중국, 일본, 미국과 어떻게 경쟁해야할지 기술하세요."
inputs = [HumanMessage(content=query)]

for event in app_reflection.stream({"messages": inputs}, stream_mode="values"):   
    message = event["messages"][-1]
    if message.content and len(event["messages"])>1:
        print('generate: ', message.content)
```

아래는 생성된 에세이 초안입니다. 

```text
한국 인공지능 산업의 발전을 위해서는 다음과 같은 준비가 필요합니다:

1. 인재 양성
인공지능 기술을 선도할 수 있는 우수한 인재 양성이 필수적입니다. 정부와 기업, 대학이 협력하여 인공지능 전문가를 체계적으로 육성하고, 해외 두뇌 유치에도 힘써야 합니다.

2. 연구개발 투자 확대
정부와 민간 기업의 인공지능 연구개발에 대한 투자를 대폭 확대해야 합니다. 기초연구와 응용연구에 균형있게 투자하여 원천기술 확보와 상용화를 동시에 추진해야 합니다.

3. 데이터 인프라 구축
인공지능 기술 발전을 위해서는 양질의 데이터 확보가 관건입니다. 정부 주도로 공공데이터를 적극 개방하고, 민간 기업의 데이터 구축을 지원해야 합니다.

4. 규제 정비 및 제도 마련
인공지능 기술의 안전성과 윤리성을 보장하기 위한 규제와 제도를 정비해야 합니다. 개인정보 보호, 알고리즘 공정성 등에 대한 기준을 마련해야 합니다.

5. 국제 협력 강화
인공지능 기술은 국가 간 경쟁이 치열한 분야입니다. 한국은 미국, 중국, 일본 등 선진국과의 협력을 강화하여 기술 교류와 공동 연구를 활성화해야 합니다.

한국은 우수한 ICT 인프라와 기술력을 바탕으로 인공지능 강국으로 도약할 수 있는 잠재력을 가지고 있습니다. 정부와 민간이 힘을 모아 체계적인 준비를 해나간다면 국제 경쟁력을 확보할 수 있을 것입니다.
```

최종적으로 완성된 에세이는 아래와 같습니다.

```python
한국은 우수한 ICT 인프라와 기술력을 바탕으로 인공지능 강국으로 도약할 잠재력을 가지고 있습니다. 하지만 미국, 중국, 일본 등 선진국과의 기술 격차를 좁히기 위해서는 다음과 같은 전략적 준비가 필요합니다.

1. 인재 양성 및 두뇌 유치
정부와 기업, 대학이 협력하여 인공지능 전문가를 체계적으로 육성해야 합니다. 또한 해외 우수 인재 유치를 위한 제도와 인센티브를 마련해야 합니다. 2021년 기준 한국의 인공지능 전문인력은 3만 명 수준으로 미국의 10분의 1에 불과합니다.

2. 연구개발 투자 확대  
정부와 민간 기업의 인공지능 연구개발 투자를 대폭 확대해야 합니다. 기초연구와 응용연구에 균형있게 투자하여 원천기술 확보와 상용화를 동시에 추진해야 합니다. 2021년 한국의 인공지능 연구개발 투자 규모는 약 1조 원으로 미국의 10분의 1 수준입니다.

3. 데이터 인프라 구축 및 개방
인공지능 기술 발전을 위해서는 양질의 데이터 확보가 관건입니다. 정부 주도로 공공데이터를 적극 개방하고, 민간 기업의 데이터 구축을 지원해야 합니다. 삼성전자, SK하이닉스 등 국내 주요 기업들이 보유한 데이터를 활용할 수 있는 방안도 모색해야 합니다.

4. 규제 정비 및 제도 마련
인공지능 기술의 안전성과 윤리성을 보장하기 위한 규제와 제도를 정비해야 합니다. 개인정보 보호, 알고리즘 공정성 등에 대한 기준을 마련하고 관련 법제를 선제적으로 정비해야 합니다. 유럽연합의 인공지능법 제정 등 국제 동향에 발맞춰 나가야 합니다.

5. 국제 협력 강화
한국은 미국, 중국, 일본 등 선진국과의 협력을 강화하여 기술 교류와 공동 연구를 활성화해야 합니다. 특히 반도체, 배터리 등 한국이 강점을 가진 분야에서 인력 교류, 공동 연구 프로젝트 등의 전략적 제휴를 모색할 필요가 있습니다.

정부와 민간이 힘을 모아 체계적인 준비를 해나간다면 한국은 인공지능 기술 경쟁에서 주도권을 확보할 수 있을 것입니다. 인공지능 기술 주도국으로 도약하면 국가 경쟁력 제고와 신산업 창출, 일자리 창출 등 많은 기회가 있을 것으로 기대됩니다.
```

LangSmith로 확인해 보면 아래와 같이 여러 단계를 거쳐서 generation/reflection이 수행되고 있음을 알 수 있습니다.

![image](https://github.com/user-attachments/assets/d40b049f-3fc3-4e26-909c-d04236b36c27)





