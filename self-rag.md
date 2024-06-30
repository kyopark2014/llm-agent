# Self RAG

[LangGraph - Self-RAG](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb?ref=blog.langchain.dev)와 같이 Self RAG는 RAG를 grade 한 후에 얻어진 결과가 환각(hallucination)을 하는지 확인하는 절차를 포함합니다. 결과가 만족하지 않을 경우에는 cycle을 통해 반복적으로 Answer를 찾습니다.

<img width="934" alt="image" src="https://github.com/kyopark2014/llm-agent/assets/52392004/d066967c-b92c-4951-973f-753d24e3e491">
