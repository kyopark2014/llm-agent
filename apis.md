# Agent

### Internet Search

- Google Search

- Tavily Search

[LangChain: Tavily Search API](https://python.langchain.com/v0.1/docs/integrations/retrievers/tavily/)와 [api-tavily-search.ipynb](./api/api-tavily-search.ipynb)을 참조합니다.

  

### Custom 함수

- 현재 날짜, 시간등의 정보 조회하기

- 시스템 시간 (한국)

[api-current-time.ipynb](./api/api-current-time.ipynb)와 같이 구현합니다.

  
## API 예제

### 도서 정보 가져오기

교보문고의 Search API를 이용하여 아래와 같이 [도서정보를 가져오는 함수](https://colab.research.google.com/drive/1juAwGGOEiz7h3XPtCFeRyfDB9hspQdHc?usp=sharing)를 정의합니다.

```python
from langchain.agents import tool
import requests
from bs4 import BeautifulSoup

@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    
    return answer
```

### 날짜와 시간 정보 가져오기

```python
@tool
def get_current_time(format: str)->str:
    """Returns the current date and time in the specified format"""
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    
    return timestr
```

### 날씨 정보 가져오기

```python
@tool
def get_weather_info(city: str) -> str:
    """
    Search weather information by city name and then return weather statement.
    city: the english name of city to search
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    
    chat = get_chat(LLM_for_chat, selected_LLM)
                
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key:
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            print('result: ', result)
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
        
    print('weather_str: ', weather_str)                            
    return weather_str
```

### Tavily Search 

아래와 같이 [Travily Search](https://app.tavily.com/home)를 이용해 검색합니다. 이를 위해서는 API Key를 미리 발급 받아서 아래와 같이 TAVILY_API_KEY로 등록하여야 합니다.

```python
os.environ["TAVILY_API_KEY"] = api_key

@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general information by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    
    answer = ""
    
    if tavily_api_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(k=5)
                    
        output = search.invoke(keyword)
        print('tavily output: ', output)
        
        for result in output[:3]:
            content = result.get("content")
            url = result.get("url")
            
            answer = answer + f"{content}, URL: {url}\n\n"
    
    return answer
```

### OpenSearch

OpenSearch를 이용해 RAG를 구성하고 필요한 정보를 검색하여 사용합니다.

```python
def search_by_opensearch(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
    
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = "idx-*", # all
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    answer = ""
    top_k = 3   
    relevant_documents = vectorstore_opensearch.similarity_search_with_score(
        query = keyword,
        k = top_k,
    )

    for i, document in enumerate(relevant_documents):
        print(f'## Document(opensearch-vector) {i+1}: {document}')

        excerpt = document[0].page_content        
        uri = document[0].metadata['uri']
                    
        answer = answer + f"{excerpt}, URL: {uri}\n\n"
    
    return answer
```



### Google Search

필요한 패키지는 아래와 같이 설치합니다.

```text
pip install google-api-python-client>=2.100.0
```

```python
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent

search = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id='a6cd8c517fbd64b72',
    k=5,
    siterestrict=False
)

google_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Use for when you need to perform an internet search to find information that another tool can not provide.",
)

search.run('langchain의 agent는 무엇이야?')
```

이때의 결과는 아래와 같습니다. 썩 좋은 결과는 아닌것으로 보여집니다.

```text
'"Olivia Wilde의 남자 친구인 Harry Styles는 29 세이고, 그의 나이에 0.23 거듭제곱한 값은 2.169459462491557 이야." 이 예제는 LangChain 문서\xa0... Jul 16, 2023 ... Langchain 이란? Langchain은 language model 기반의 ... 사실 간단한 챗이야 ... AgentType 을 보면 감이 오시겠지만, 우리는 다양한 종류의 Agent를\xa0... Agent 활용에서는 LangChain의 ReAct Agent를 정의합니다. ... "엔씨의 Lex 서비스는 무엇인지 설명해줘."와 같이 ... 메뉴에서 "Timestamp Extraction"을 선택하고, "지금은\xa0... Aug 16, 2023 ... ... 의 답변을 받을 수는 있어도 상세한 정보를 얻을 수는 없다. 이러한 한계를 극복하기 위해 LangChain에서는 검색을 통해 언어모델에 지식을 보완하는\xa0... Jul 3, 2023 ... 현재 LangChain에서 사용되는 에이전트 체인은 사용자와 AI의 요청과 응답으로 구성된 구성된 프롬프트 처리를 지원하지 않습니다. 우리는 주로 이를 모델\xa0...'
```
