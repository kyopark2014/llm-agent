# LangChain의 Agent 사용하기

여기서는 LangChain의 [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)를 이용하여 Agent를 정의합니다.

## Tool calling agent




## ReAct










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

