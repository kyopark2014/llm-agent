# LangChain의 Agent 사용하기

LLM을 사용할때 다양한 API로 부터 얻은 결과를 사용하여 더 정확한 결과를 얻고 싶을때에 Agent를 사용합니다. Agent를 사용하는 대표적인 Architecture는 아래와 같습니다. 

1) 사용자가 채팅창에서 질문을 입력하면 WebSocket 방식으로 Lambda(chat)에 전달됩니다.
2) Lambda(chat)은 Agent 동작을 수행하는데, Action - Observation - Thought - Final Answer의 동작을 수행합니다. 만약 Thought에서 Final Answer를 얻지 못하면 Action을 추가 수행합니다.
3) Agent에서 Action을 수행하기 위하여 API를 이용해 필요한 정보를 얻어옵니다. 이때 사용하는 API에는 도서 추천, 날씨정보, 검색엔진이 있을 수 있습니다. 또한 시스템 시간을 가져오는 동작은 별도 API가 아닌 내부 함수를 이용해 구현할 수 있습니다.
4) 만약 RAG의 정보가 필요한 경우에는 Action의 하나로 RAG를 구현한 OpenSearch를 통해 필요한 정보를 조회합니다.
5) Obseravation/Thought/Final Answer를 위해 Agent는 prompt를 이용해 LLM에 요청을 보내고 응답을 받습니다.
6) Agent가 Final Answer를 구하면 사용자에게 전달합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/c372c125-4e05-41f8-b691-784e4c2028af)


## Agent의 동작

Agent의 동작은 Action, Observation, Thought와 같은 동작을 반복적으로 수행하여 Final Answer를 얻습니다. Agent에 대한 자세한 내용은 [agent.md](./agent.md)을 참조합니다.

1) LLM으로 Tool들로 부터 하나의 Action을 선택합니다. 이때에는 tool의 description을 이용합니다.
2) Action을 수행합니다.
3) Action결과를 관찰(Observation)합니다.
4) 결과가 만족스러운지 확인(Thought) 합니다. 만족하지 않으면 반복합니다.

[LLM-powered autonomous agent system](https://lilianweng.github.io/posts/2023-06-23-agent/)으로 부터 구현된 Agent는 다음과 같습니다. 


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

- 오늘 날짜를 알수 있으므로 "올해 크리스마스까지 몇일이 남아 있어?"와 같은 질문에 정확히 답변할 수 있습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/8905c677-7a26-4a4e-9e14-ee8af8a481cf)


- "서울과 부산의 날씨를 알려줘"와 같이 서울과 부산의 결과를 모두 기대하고 입력시에 아래와 같은 결과를 얻습니다. 

<img width="848" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/7b5c4993-1178-442d-9fb0-ddaff6b7ab09">

이때의 LangSmith의 로그를 확인하면 서울과 부산과 대한 검색후 결과를 생성하였습니다. (get_weather_info를 서울과 부산에 대해 각각 호출함)

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/38334666-c71d-4076-9be1-eb8fc16a34f5)


- "미국 여행을 하려고 해. 추천해줘 어떻게 여행하는게 좋아?"로 질문을 하면 아래와 같이 로스웬젤레스를 추천해주는데 날씨정보도 같이 전달하고 있습니다.

상세한 내부 동작은 아래와 같습니다. 

1) 질문에 필요한 정보를 찾습니다. 여기에서는 여행일정, 방문도시, 관심사에 선택했습니다.
2) 현재 가지고 있는 api중에 관련된 것을 찾았는데, 도서정보를 찾는 API(get_product_list)가 선택되었습니다.
3) "미국 여행 가이드 북"을 검색해서 도서 정보를 얻었습니다.
4) 가이드된 도서 목록에서 미국 서부 지역, 뉴욕을 선택하고 추가 정보를 찾으려고 합니다.
5) API중에 get_weather_info을 선택해서, Los Angeles 를 검색합니다.
6) 가이드 북과 날씨 정보를 조합하여 Final Answer로 여행 계획을 추천합니다. 

아래는 LangSmith에서 추출한 로그 입니다. 

```text
Question: 미국 여행을 하려고 해. 추천해줘 어떻게 여행하는게 좋아?
Thought:Thought: 미국 여행에 대한 구체적인 정보가 필요할 것 같습니다. 여행 일정, 방문 도시, 관심사 등에 대해 더 자세히 알아보는 것이 좋겠습니다.
Action: get_product_list
Action Input: 미국 여행 가이드북
Observation: 추천 도서는 아래와 같습니다.
[국내도서]미국 서부 셀프트래블(2022-2023), URL: https://product.kyobobook.co.kr/detail/S000060626443
[eBook]미국 서부 셀프트래블(2022-2023), URL: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000002960164
[국내도서]버킷리스트 미국 서부, URL: https://product.kyobobook.co.kr/detail/S000001981279
[eBook]살아보고 싶다면, 포틀랜드, URL: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000003234513
[국내도서]뉴욕 영화 가이드북, URL: https://product.kyobobook.co.kr/detail/S000001913689

Thought: Thought: 여행 가이드북 목록을 보니 미국 서부 지역과 뉴욕에 대한 정보가 있습니다. 하지만 구체적인 여행 계획을 세우기 위해서는 더 자세한 정보가 필요할 것 같습니다.
Action: get_weather_info
Action Input: Los Angeles
Observation: 로스앤젤레스의 현재 날씨의 특징은 Clouds이며, 현재 온도는 13.85도 이고, 최저온도는 11.85도, 최고 온도는 15.6도 입니다. 현재 습도는 85% 이고, 바람은 초당 2.06 미터 입니다. 구름은 100% 입니다.
Thought: 미국 여행에 대한 구체적인 정보를 모으기 위해 여러 도구를 사용해 보았습니다. 가이드북 목록을 통해 인기 여행지와 관심 분야를 파악할 수 있었고, 날씨 정보를 통해 여행 시기와 준비물을 고려할 수 있었습니다. 이제 이 정보들을 종합하여 여행 계획을 세울 수 있을 것 같습니다.

Final Answer: 미국 여행을 계획할 때는 다음 사항들을 고려하는 것이 좋습니다.

1. 여행 지역 선정: 가이드북 목록에서 보았듯이 서부 지역(로스앤젤레스, 포틀랜드 등)과 뉴욕이 인기 여행지입니다. 관심사와 일정에 맞춰 방문 도시를 정하세요.

2. 계절 및 날씨: 로스앤젤레스 날씨 정보를 보니 현재 구름이 많고 온도는 10도 중반입니다. 여행 시기에 따라 날씨가 다르므로 이에 맞춰 옷차림과 활동을 계획하는 것이 좋습니다. 

3. 교통 및 숙박: 렌터카 이용이 편리할 수 있습니다. 도시 간 이동 시에는 기차나 항공편을 이용할 수 있습니다. 숙박 시설은 호텔, 모텔, 에어비앤비 등 다양한 선택지가 있습니다.

4. 관광명소 및 활동: 뉴욕 영화 가이드북에서 보듯 영화 촬영지 방문, 미술관 관람 등 취향에 맞는 활동을 계획하세요.
```

이때의 결과는 아래와 같습니다. 

<img width="858" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/75987226-b3d2-481c-a9ba-efc62bfbe4ca">


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
