# LangChain의 Agent 사용하기

[LangChain의 Agent Type](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)을 보면, [Tool Calling](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/), [OpenAI tools](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/), [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)가 있습니다. 

ReAct의 경우에 직관적이고 이해가 쉬운 반면에 Multi-Input Tools, Parallel Function Calling과 같은 기능을 제공하지 않고 있습니다. 반면에 OpenAI tools는 가장 많이 사용되고 있고, 다양한 사례를 가지고 있습니다. Tool Calling은 OpenAI tools와 유사한 방식으로 Anthropic, Gemini등을 지원하고 있습니다. 여기에서는 Bedrock의 Anthropic Model을 이용하여 Agent를 구성합니다. 따라서, ReAct와 Tool calling agent를 모두 설명합니다. 

## ReAct

LangChain의 [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)를 이용하여 Agent를 정의합니다.


## Tool calling agent

[Tool calling agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)








## Reference

[Intro to LLM Agents with Langchain: When RAG is Not Enough](https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834)

[LangChain 🦜️🔗 Tool Calling and Tool Calling Agent 🤖 with Anthropic](https://medium.com/@dminhk/langchain-%EF%B8%8F-tool-calling-and-tool-calling-agent-with-anthropic-467b0fb58980)

[Automating tasks using Amazon Bedrock Agents and AI](https://blog.serverlessadvocate.com/automating-tasks-using-amazon-bedrock-agents-and-ai-4b6fb8856589)

