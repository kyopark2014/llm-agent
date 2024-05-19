# Agent

## Lambda Agent

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


## Tavily Search 사용 예

[Teddylee 가이드](https://teddylee777.github.io/langchain/langchain-agent/)

```python
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(k=5)

search.invoke("판교 카카오 프렌즈샵 아지트점의 전화번호는 무엇인가요?")

tools = [search, retriever_tool]

tools = [search, retriever_tool]

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "안녕, 반가워!"})
print(f'답변: {response["output"]}')
```
