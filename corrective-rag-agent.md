# Corrective RAG Agent

Self Reflection을 이용해 RAG의 성능을 향상시킵니다.

[langgraph_crag_mistral.ipynb](https://github.com/mistralai/cookbook/blob/main/third_party/langchain/langgraph_crag_mistral.ipynb)에서는 Self Reflection을 이용해 RAG의 성능을 강화합니다.

![image](https://github.com/kyopark2014/llm-agent/assets/52392004/3a2618d0-0e81-4900-976e-78d30fd19a0e)


아래와 같이 Graph를 생성합니다.

```python
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
```
