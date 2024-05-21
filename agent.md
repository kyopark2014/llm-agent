# Agent

## Reasoning

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
