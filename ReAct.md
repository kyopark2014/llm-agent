# ReAct

[ReAct Prompting](https://www.promptingguide.ai/techniques/react)와 같이 ReAct는 reasoning traces와 task-specific actions을 교차 배열하는 방식(interleaved manner)으로 동작합니다. 

## 동작 방식

[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)에서는 LLM에게 단계별로 추론 흔적(reseoning traces)과 작업별 조치들(task-specific actions)을 생성하도록 요청하면 작업 수행 능력이 향상된다(the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner) 라고 설명하고 있습니다.

아래와 같이 [LangChain의 ReAct는 Reasoning와 Action](https://www.youtube.com/watch?v=Eug2clsLtFs)으로 표현됩니다. 여기서는 생각하는 방법(Paradigm of thinking)으로 Reseoning은 답변의 정당성(justification of the answer)을 확인하여, 답변에 필요한 것(prime)을 찾습니다. 또한, 현재의 환경(Envrionment)에서 어떤 행동(Action)을 선택(SayCan)하고, 관찰(Obseravation)을 통해 결과를 확인합니다.  

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/f75501fd-d3d5-4a3f-9b6c-42b5466bb3f9)


동작 방식을 이해하기 위하여 [YT_LangChain_Agents.ipynb](https://github.com/samwit/langchain-tutorials/blob/main/agents/YT_LangChain_Agents.ipynb)와 같이 agent를 실행합니다.

```text
agent.run("Who is the United States President? What is his current age raised divided by 2?")
```

이때의 실행 결과는 아래와 같습니다.

```text
> Entering new AgentExecutor chain...
 I need to find out who the President is and then do some math.
Action: Search
Action Input: "United States President"
Observation: Joe Biden
Thought: I now need to find out Joe Biden's age.
Action: Search
Action Input: "Joe Biden age"
Observation: 80 years
Thought: I now need to divide Joe Biden's age by 2.
Action: Calculator
Action Input: 80/2
Observation: Answer: 40
Thought: I now know the final answer.
Final Answer: Joe Biden is the United States President and his current age raised divided by 2 is 40.
```


[hello-langchain-6.py](https://github.com/chrishayuk/how-react-agents-work/blob/main/hello-langchain-6.py)


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


