# LangChain의 Agent 사용하기

LLM을 사용할때 다양한 API로 부터 얻은 결과를 사용하여 더 정확한 결과를 얻고 싶을때에 Agent를 사용합니다.

## Agent의 동작

Agent의 동작은 Action, Observation, Thought와 같은 동작을 반복적으로 수행하여 Final Answer를 얻습니다. Agent에 대한 자세한 내용은 [agent.md](./agent.md)을 참조합니다.

1) LLM으로 Tool들로 부터 하나의 Action을 선택합니다. 이때에는 tool의 description을 이용합니다.
2) Action을 수행합니다.
3) Action결과를 관찰(Observation)합니다.
4) 결과가 만족스러운지 확인(Thought) 합니다. 만족하지 않으면 반복합니다.



## Agent Type

LangChain의 [Agent Type](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)을 보면, [Tool Calling](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/), [OpenAI tools](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/), [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)가 있습니다. 

- ReAct의 경우에 직관적이고 이해가 쉬운 반면에 Multi-Input Tools, Parallel Function Calling과 같은 기능을 제공하지 않고 있습니다.
- OpenAI tools는 가장 많이 사용되고 있고, 다양한 사례를 가지고 있습니다.
- Tool Calling은 OpenAI tools와 유사한 방식으로 Anthropic, Gemini등을 지원하고 있습니다.

여기에서는 ReAct와 Tool calling agent에 대해 설명합니다. 

### ReAct

LangChain의 [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)를 이용하여 Agent를 정의합니다. ReAct에 대한 자세한 내용은 [ReAct.md](./ReAct.md)을 참조합니다. 

최종 답변에 한 번에 도달하는 대신에 여러 단계의 thought(사고)-action(행동)-observation(관찰) 과정을 통해 과제를 해결할 수 있고, 환각도 줄일 수 있습니다. 이때의 한 예는 아래와 같습니다. 

```text
Thought -> Action (Search) -> Observation -> Thought - Action (Search) -> Observation -> Thought -> Final Result
```

실제 실행한 결과는 아래와 같습니다.

- "오늘 날씨 알려줘"를 입력하면 현재의 [날씨 정보를 조회](./apis.md#%EB%82%A0%EC%94%A8-%EC%A0%95%EB%B3%B4-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)하여 알려줍니다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4b2f79cc-6782-4c44-b594-1c5f22472dc7)

- "오늘 날짜 알려줘"를 하면 [시스템 날짜를 확인](./apis.md#%EB%82%A0%EC%A7%9C%EC%99%80-%EC%8B%9C%EA%B0%84-%EC%A0%95%EB%B3%B4-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)하여 알려줍니다. 

<img width="850" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/a0190426-33d4-46d3-b9d2-5294f9222b8c">

- "서울 여행에 대한 책을 추천해줘"를 입력하면 [교보문의 책검색 API](./apis.md#%EB%8F%84%EC%84%9C-%EC%A0%95%EB%B3%B4-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0)를 이용하여 관련책을 검색하여 추천합니다.

<img width="849" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/e62b4654-ba18-40e6-86ae-2152b241aa04">


- "서울과 부산의 날씨를 알려줘"와 같이 서울과 부산의 결과를 모두 기대하고 입력시에 아래와 같은 결과를 얻습니다. 

<img width="848" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/7b5c4993-1178-442d-9fb0-ddaff6b7ab09">

이때의 LangSmith의 로그를 확인하면 서울과 부산과 대한 검색후 결과를 생성하였습니다. (get_weather_info를 서울과 부산에 대해 각각 호출함)

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/38334666-c71d-4076-9be1-eb8fc16a34f5)


### Tool calling agent

LangChain의 [Tool calling agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)은 Multi-Input Tools, Parallel Function Calling와 같은 다양한 기능을 제공하고 있습니다. 상세한 내용은 [toolcalling.md](https://github.com/kyopark2014/llm-agent/blob/main/toolcalling.md)을 참조합니다. 

- [Chat models](https://python.langchain.com/v0.1/docs/integrations/chat/)에 따르면, BedrockChat은 Tool calling agent을 지원하고 있지 않습니다.
- ChatBedrock API는 Agent를 선언할 수 있으나, 아래의 LangSmith 결과와 같이 Tool Calling에 대한 응답을 얻지 못하고 있습니다.
- Tool calling은 ReAct에서 지원하지 못하고 있는 Multi-Input Tools, Parallel Function Calling등을 지원하고 있으므로, 향후 지원을 기대해 봅니다.
  
![image](https://github.com/kyopark2014/llm-agent/assets/52392004/86364b1b-0f52-4faa-b370-dd6660d4974f)

## Prompt 

ReAct를 위한 Prompt 에제는 [prompt.md](./prompt.md)을 참조합니다.

## 외부 API 

[apis.md](./apis.md)에서는 도서 검색, 날씨, 시간과 같은 유용한 검색 API에 대해 설명하고 있습니다.

## LangSmith 사용 설정

[langsmith.md](./langsmith.md)은 [LangSmith](https://smith.langchain.com/)에서 발급한 api key를 설정하여, agent의 동작을 디버깅할 수 있도록 해줍니다. 

## LLM의 선택

Agent 사용시 Tool을 선택하고, Observation과 Thought을 통해 Action으로 얻어진 결과가 만족스러운지 확인하는 과정이 필요합니다. 따라서 LLM의 성능은 Agent의 결과와 밀접한 관계가 있습니다. Claude Sonnet으로 Agent를 만든 결과가 일반적으로 Claude Haiku보다 우수하여, Sonnet을 추천합니다.

## Reference

[Using LangChain ReAct Agents for Answering Multi-hop Questions in RAG Systems](https://towardsdatascience.com/using-langchain-react-agents-for-answering-multi-hop-questions-in-rag-systems-893208c1847e)

[Intro to LLM Agents with Langchain: When RAG is Not Enough](https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834)

[LangChain 🦜️🔗 Tool Calling and Tool Calling Agent 🤖 with Anthropic](https://medium.com/@dminhk/langchain-%EF%B8%8F-tool-calling-and-tool-calling-agent-with-anthropic-467b0fb58980)

[Automating tasks using Amazon Bedrock Agents and AI](https://blog.serverlessadvocate.com/automating-tasks-using-amazon-bedrock-agents-and-ai-4b6fb8856589)

[llama3 로 #agent 🤖 만드는 방법 + 8B 오픈 모델로 Agent 구성하는 방법](https://www.youtube.com/watch?app=desktop&v=04MM0PXv2Fk)
