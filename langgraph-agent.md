# LangGraph Agent

## Graph state

- input: 사용자로부터 입력으로 전달된 주요 요청을 나타내는 입력 문자열
- chat_history: 이전 대화 메시지
- intermediate_steps: Agent가 시간이 지남에 따라 취하는 행동과 관찰 사항의 목록.
- agent_outcome: Agent의 응답. AgentAction인 경우에 tool을 호출하고, AgentFinish이면 AgentExecutor를 종료함

상세한 내용은 [agent_executor/base.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb)을 참조합니다.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
```
