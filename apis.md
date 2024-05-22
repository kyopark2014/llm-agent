# Agent

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

### Lambda Agent

[LangChain Agent - AWS Lambda](https://python.langchain.com/v0.1/docs/integrations/tools/awslambda/)를 참조합니다.

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)

tools = load_tools(
    ["awslambda"],
    awslambda_tool_name="email-sender",
    awslambda_tool_description="sends an email with the specified content to test@testing123.com",
    function_name="testFunction1",
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("Send an email to test@testing123.com saying hello world.")
```


### Tavily Search 사용 예

[Teddylee 가이드](https://teddylee777.github.io/langchain/langchain-agent/)

[Travily](https://wikidocs.net/234282)

```python
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(k=5)

search.invoke("판교 카카오 프렌즈샵 아지트점의 전화번호는 무엇인가요?")

tools = [search, retriever_tool]

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "안녕, 반가워!"})
print(f'답변: {response["output"]}')
```

### Google Search

[Langchain agent 내부 동작 구조 이해](https://bcho.tistory.com/m/1427)

```python
from langchain.llms.openai import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
import os

google_search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search",
        verbose=True
    )
]

template = '''Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

search_agent = create_react_agent(model,tools,prompt)
agent_executor = AgentExecutor(
    agent=search_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
response = agent_executor.invoke({"input": "Where is the hometown of the 2007 US PGA championship winner and his score?"})
print(response)
```





