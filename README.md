# LLM Agent

# LangChain의 Agent 사용하기

## 외부 검색 API 

### 도서 정보 가져오기

교보문고의 Search API를 이용하여 아래와 같이 [도서정보를 가져오는 함수](https://colab.research.google.com/drive/1juAwGGOEiz7h3XPtCFeRyfDB9hspQdHc?usp=sharing)를 정의합니다.

```python
from langchain.agents import tool
import requests
from bs4 import BeautifulSoup

@tool
def get_product_list(keyword: str) -> list:
    """
    Search product list by keyword and then return product list
    keyword: search keyword
    return: product list
    """

    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        prod_list = [
            {"title": prod.text.strip(), "link": prod.get("href")} for prod in prod_info
        ]
        return prod_list[:5]
    else:
        return []
```

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

## React

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

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
```



## 동작 설명

아래는 CloudWatch에서 읽어온 실행 로그입니다. AgentExecutor chain이 동작하면서 먼저 Thought로 여행 관련 도서 검색을 필요하다는것을 인지하면, get_product_list 함수를 이용하여 "여행"을 검색하고 결과를 이용해 답변하게 됩니다.

```text
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3mThought: 이 질문에 대한 답변을 하기 위해서는 여행 관련 도서 목록을 검색해야 합니다.
Action: get_product_list
Action Input: 여행[0m
[33;1m[1;3m[{'title': '[국내도서]\n예약판매\n우리문학의여행.다문화.디아스포라', 'link': 'https://product.kyobobook.co.kr/detail/S000213330319'}, {'title': '[국내도서]\n예약판매\n해시태그 프랑스 소도시여행', 'link': 'https://product.kyobobook.co.kr/detail/S000213329696'}, {'title': '[국내도서]\n예약판매\n지도 위 쏙쏙 세계여행 액티비티북 프랑스', 'link': 'https://product.kyobobook.co.kr/detail/S000213325676'}, {'title': '[국내도서]\n예약판매\n혼자서 국내 여행(2024~2025 최신판)', 'link': 'https://product.kyobobook.co.kr/detail/S000213304266'}, {'title': '[국내도서]\n예약판매\n친구랑 함께한 세계여행', 'link': 'https://product.kyobobook.co.kr/detail/S000213290121'}][0m
```



## Reference

[Automating tasks using Amazon Bedrock Agents and AI](https://blog.serverlessadvocate.com/automating-tasks-using-amazon-bedrock-agents-and-ai-4b6fb8856589)

