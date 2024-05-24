# LangChain의 Agent 사용하기

LLM을 사용할때 다양한 API로 부터 얻은 결과를 사용하여 더 정확한 결과를 얻고 싶을때에 Agent를 사용합니다.

## Agent의 동작

Agent의 동작은 Action, Observation, Thought와 같은 동작을 반복적으로 수행하여 Final Answer를 얻습니다. Agent에 대한 자세한 내용은 [agent.md](./agent.md)을 참조합니다.

1) LLM으로 Tool들로 부터 하나의 Action을 선택합니다. 이때에는 tool의 description을 이용합니다.
2) Action을 수행합니다.
3) Action결과를 관찰(Obseravation)합니다.
4) 결과가 만족스러운지 확인(Thought) 합니다. 만족하지 않으면 반복합니다.



## Agent Type

LangChain의 [Agent Type](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)을 보면, [Tool Calling](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/), [OpenAI tools](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/), [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)가 있습니다. ReAct의 경우에 직관적이고 이해가 쉬운 반면에 Multi-Input Tools, Parallel Function Calling과 같은 기능을 제공하지 않고 있습니다. 반면에 OpenAI tools는 가장 많이 사용되고 있고, 다양한 사례를 가지고 있습니다. Tool Calling은 OpenAI tools와 유사한 방식으로 Anthropic, Gemini등을 지원하고 있습니다. 여기에서는 Bedrock의 Anthropic Model을 이용하여 Agent를 구성합니다. 따라서, ReAct와 Tool calling agent를 모두 설명합니다. 

### ReAct

LangChain의 [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)를 이용하여 Agent를 정의합니다. 최종 답변에 한 번에 도달하는 대신에 여러 단계의 thought(사고)-action(행동)-observation(관찰) 과정을 통해 과제를 해결할 수 있고, 환각도 줄일 수 있습니다. 이때의 한 예는 아래와 같습니다. 

Thought -> Action (Search) -> Observation -> Thought - Action (Search) -> Observation -> Thought -> Final Result

실제 실행한 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/4b2f79cc-6782-4c44-b594-1c5f22472dc7)

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/69ff3e46-ec3e-4ba1-9f10-380b31554f15)


[Using LangChain ReAct Agents for Answering Multi-hop Questions in RAG Systems](https://towardsdatascience.com/using-langchain-react-agents-for-answering-multi-hop-questions-in-rag-systems-893208c1847e)


ReAct에 대한 자세한 내용은 [ReAct.md](./ReAct.md)을 참조합니다.


### Tool calling agent

[Tool calling agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)


- [Chat models](https://python.langchain.com/v0.1/docs/integrations/chat/)에서는 BedrockChat만 있고 미지원으로 표시가 되어 있습니다.
- ChatBedrock으로 테스트시에 Tool Calling은 아래와 같이 응답을 얻지 못하므로 현재 미지원이나 향후 지원이 예상됩니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/86364b1b-0f52-4faa-b370-dd6660d4974f)

## Prompt 

ReAct를 위한 Prompt 에제는 [prompt.md](./prompt.md)을 참조합니다.


## 외부 API 

[apis.md](./apis.md)에서는 도서 검색, 날씨, 시간과 같은 유용한 검색 API에 대해 설명하고 있습니다.



## LangSmith 사용 설정

[langsmith.md](./langsmith.md)와 같이 lambda-chat-ws의 envrionment의 langsmith_api_key에 [LangSmith](https://smith.langchain.com/)에서 발급한 api key를 넣으면 LangSmith로 agent의 동작을 디버깅할 수 있습니다. 


## Reference

[Intro to LLM Agents with Langchain: When RAG is Not Enough](https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834)

[LangChain 🦜️🔗 Tool Calling and Tool Calling Agent 🤖 with Anthropic](https://medium.com/@dminhk/langchain-%EF%B8%8F-tool-calling-and-tool-calling-agent-with-anthropic-467b0fb58980)

[Automating tasks using Amazon Bedrock Agents and AI](https://blog.serverlessadvocate.com/automating-tasks-using-amazon-bedrock-agents-and-ai-4b6fb8856589)

[llama3 로 #agent 🤖 만드는 방법 + 8B 오픈 모델로 Agent 구성하는 방법](https://www.youtube.com/watch?app=desktop&v=04MM0PXv2Fk)
