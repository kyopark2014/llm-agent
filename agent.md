# Agent

## Reference

[LangChain Agents & LlamaIndex Tools](https://cobusgreyling.medium.com/langchain-agents-llamaindex-tools-e74fd15ee436)에서는 아래와 같은 cycle을 설명하고 있습니다. 

- 어떤 요청(request)를 받았을때 agent는 LLM이 하려고 하는 어떤 action을 할지 결정할때 이용된다.

- Action이 완료된 후에 Observation하고 이후에 Thought에서는 Final Answer에 도달한지 확인한다. Final answer가 아니라면 다른 action을 수행하는 cycle을 거친다. 

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/6b2032db-c259-43f3-a699-7eca41117d45)


