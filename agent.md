# Agent

LLM을 사용할때 다양한 API로 부터 얻은 결과를 사용하고 싶을때 Agent를 사용합니다.

1) LLM으로 Tool들로 부터 하나의 Action을 선택합니다. 이때에는 tool의 description을 이용합니다.
2) Action을 수행합니다.
3) Action결과를 관찰(Obseravation)합니다.
4) 결과가 만족스러운지 확인(Thought) 합니다. 만족하지 않으면 반복합니다.

## Agent의 정의

아래와 같이 Agent를 ReAct로 정의합니다. 결과는 아래와 같이 stream으로 출력합니다.

```python
from langchain.agents import AgentExecutor, create_react_agent

def use_agent(connectionId, requestId, chat, query):
    tools = [check_system_time, get_product_list]
    prompt_template = get_react_prompt_template()
    print('prompt_template: ', prompt_template)
    
    agent = create_react_agent(chat, tools, prompt_template)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    response = agent_executor.invoke({"input": query})
    print('response: ', response)
    
    msg = readStreamMsg(connectionId, requestId, response['output'])

    msg = response['output']
    print('msg: ', msg)
            
    return msg
```

## Prompt

[Agent Concept](https://python.langchain.com/v0.1/docs/modules/agents/concepts/)을 참조합니다.

### ReAct

[Reasoning and Action의 약자](https://blog.kubwa.co.kr/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-%EB%9E%AD%EC%B2%B4%EC%9D%B8%EA%B4%80%EB%A0%A8-%EB%85%BC%EB%AC%B8-react-synergizing-reasoning-and-acting-in-language-models-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%8B%A4%EC%8A%B5-w-pytorch-dd31321ead00)로서, reasoning trace는 CoT(Chain of Thought)을 기초로 하고, Reasoning과 action을 반복적으로 수행하면서 환각(Hallucination)과 에러 전파(error properation)을 줄일 수 있습니다. 이를 통해 사람처럼 task를 푸는 것(human like task solving trajectory)을 가능하게 합니다.

이때, ReAct를 위한 Prompt는 [hwchase17/react](https://smith.langchain.com/hub/hwchase17/react)을 이용해 아래와 같이 정의합니다.

```python
def get_react_prompt_template():
    # Get the react prompt template
    return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
```

#### 동작 설명

아래는 CloudWatch에서 읽어온 실행 로그입니다. AgentExecutor chain이 동작하면서 먼저 Thought로 여행 관련 도서 검색을 필요하다는것을 인지하면, get_product_list 함수를 이용하여 "여행"을 검색하고 결과를 이용해 답변하게 됩니다.

```text
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3mThought: 이 질문에 대한 답변을 하기 위해서는 여행 관련 도서 목록을 검색해야 합니다.
Action: get_product_list
Action Input: 여행[0m
[33;1m[1;3m[{'title': '[국내도서]\n예약판매\n우리문학의여행.다문화.디아스포라', 'link': 'https://product.kyobobook.co.kr/detail/S000213330319'}, {'title': '[국내도서]\n예약판매\n해시태그 프랑스 소도시여행', 'link': 'https://product.kyobobook.co.kr/detail/S000213329696'}, {'title': '[국내도서]\n예약판매\n지도 위 쏙쏙 세계여행 액티비티북 프랑스', 'link': 'https://product.kyobobook.co.kr/detail/S000213325676'}, {'title': '[국내도서]\n예약판매\n혼자서 국내 여행(2024~2025 최신판)', 'link': 'https://product.kyobobook.co.kr/detail/S000213304266'}, {'title': '[국내도서]\n예약판매\n친구랑 함께한 세계여행', 'link': 'https://product.kyobobook.co.kr/detail/S000213290121'}][0m
```




### ReAct Chat

[react-chat](https://smith.langchain.com/hub/hwchase17/react-chat)을 이용하여 채팅이력이 포함된 ReAct Agent를 정의할 수 있습니다.

```python
def get_react_chat_prompt_template():
    # Get the react chat prompt template
    return PromptTemplate.from_template("""Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
```

### Schema

#### AgentAction

- tool: tool의 이름
- tool_input: tool의 input

#### AgentFinish

Agent로 부터의 final result를 의미합니다.

return_values: final agent output을 포함하고 있는 key-value. output key를 가지고 있습니다. 

#### Intermediate Steps

이전 Agent action을 나타내는것으로 CURRENT agent의 실행으로 인한 output을 포함하고 있습니다. List[Tuple[AgentAction Any]]로 구성됩니다. 


  

## Tool의 종류

### Internet Search

- Google Search

- Tavily Search

[LangChain: Tavily Search API](https://python.langchain.com/v0.1/docs/integrations/retrievers/tavily/)와 [api-tavily-search.ipynb](./api/api-tavily-search.ipynb)을 참조합니다.

  

### Custom 함수

- 현재 날짜, 시간등의 정보 조회하기

- 시스템 시간 (한국)

[api-current-time.ipynb](./api/api-current-time.ipynb)와 같이 구현합니다.
  

### RAG의 Knowledge store를 이용 (Retriever)

[14-Agent/04-Agent-with-various-models.ipynb](https://github.com/teddylee777/langchain-kr/blob/main/14-Agent/04-Agent-with-various-models.ipynb)을 참조합니다.

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="2023년 12월 AI 관련 정보를 PDF 문서에서 검색합니다. '2023년 12월 AI 산업동향' 과 관련된 질문은 이 도구를 사용해야 합니다!",
)
```

### DB query


## Reasoning 방식

### Reasoning의 정의 

Reasoning in Artificial Intelligence refers to the process by which AI systems analyze information, make inferences, and draw conclusions to solve problems or make decisions. It is a fundamental cognitive function that enables machines to mimic human thought processes and exhibit intelligent behavior.



## Reference

[Agents with local models](https://www.youtube.com/watch?app=desktop&v=04MM0PXv2Fk)


[LangChain Agents & LlamaIndex Tools](https://cobusgreyling.medium.com/langchain-agents-llamaindex-tools-e74fd15ee436)에서는 아래와 같은 cycle을 설명하고 있습니다. 

- 어떤 요청(request)를 받았을때 agent는 LLM이 하려고 하는 어떤 action을 할지 결정할때 이용된다.

- Action이 완료된 후에 Observation하고 이후에 Thought에서는 Final Answer에 도달한지 확인한다. Final answer가 아니라면 다른 action을 수행하는 cycle을 거친다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/6b2032db-c259-43f3-a699-7eca41117d45)


[Introducing LangChain Agents: 2024 Tutorial with Example](https://brightinventions.pl/blog/introducing-langchain-agents-tutorial-with-example/)

- Agent는 언어 모델을 이용하여 일련의 action(sequence of actions)들을 선택한다. 여기서 Agent는 결과를 얻기 위하여 action들을 결정하는데 reasoning engine을 이용하고 있다.

- Agent는 간단한 자동 응답(automated response)로 부터 복잡한(complex), 상황인식(context-aare)한 상호연동(interaction)하는 task들을 처리하는데(handling) 중요하다.

- Agent는 Tools, LLM, Prompt로 구성된다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/e0ab693a-1b7b-492d-a19c-30dd4dddded1)

Tool에는 아래와 같은 종류들이 있습니다. 

- Web search tool: Google Search, Tavily Search, DuckDuckGo
- Embedding search in vector database
- 인터넷 검색, reasoning step
- API integration tool
- Custom Tool

#### Agent와 Chain의 차이점

Chain은 연속된 action들로 hardcoding되어 있어서 다른 path를 쓸수 없습니다. 즉, agent는 관련된 정보를 이용하여 결정을 할 수 있고, 원하는 결과를 얻을때까지 반복적으로 다시 할 수 있습니다.


![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c746c149-ecee-48fa-9c0c-ce66d03c4f34)
