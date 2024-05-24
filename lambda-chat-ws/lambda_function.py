import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import re
import traceback
import requests

from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock

from langchain.agents import tool
from langchain.agents import AgentExecutor, create_react_agent
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pytz import timezone

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
bedrock_region = os.environ.get('bedrock_region', 'us-west-2')
modelId = os.environ.get('model_id', 'amazon.titan-tg1-large')
print('model_id[:9]: ', modelId[:9])
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
debugMessageMode = os.environ.get('debugMessageMode', 'false')
agentLangMode = 'kor'

import os
langsmith_api_key = os.environ.get('LANGCHAIN_API_KEY')
print('langsmith_api_key: ', langsmith_api_key)
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

# api key to get weather information in agent
secretsmanager = boto3.client('secretsmanager')
try:
    get_secret_value_response = secretsmanager.get_secret_value(
        SecretId='openweathermap'
    )
    #print('get_secret_value_response: ', get_secret_value_response)
    secret = json.loads(get_secret_value_response['SecretString'])
    #print('secret: ', secret)
    api_key = secret['api_key']

except Exception as e:
    raise e
   
# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

def initiate_chat():
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }            
        )
    )

    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"
    def get_parameter(modelId):
        if modelId == 'amazon.titan-tg1-large' or modelId == 'amazon.titan-tg1-xlarge': 
            return {
                "maxTokenCount":1024,
                "stopSequences":[],
                "temperature":0,
                "topP":0.9
            }
        elif modelId[:9] == 'anthropic':
            return {
                "max_tokens":4096,
                "temperature":0.1,
                "top_k":250,
                "top_p": 0.9,
                "stop_sequences": [HUMAN_PROMPT]            
            }
    parameters = get_parameter(modelId)

    """
    chat = BedrockChat(
        model_id=modelId,
        client=boto3_bedrock, 
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs=parameters,
    )  
    """

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    return chat

chat = initiate_chat()

map_chain = dict() 
MSG_LENGTH = 100

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            print('text: ', text)
            print('msg: ', msg)        

            chat_memory.save_context({"input": text}, {"output": msg})             

def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

@tool 
def get_product_list(keyword: str) -> str:
    """
    Search product list by keyword and then return product list
    keyword: search keyword
    return: product list
    """

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            # \n문자를 replace합니다.
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n"
    
    return answer
    
@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    # print('timestr:', timestr)
    
    return timestr

def get_lambda_client(region):
    return boto3.client(
        service_name='lambda',
        region_name=region
    )

@tool    
def get_system_time() -> list:
    """
    retrive system time to earn the current date and time.
    return: a string of date and time
    """    
    
    function_name = "lambda-datetime-for-llm-agent"
    lambda_region = 'ap-northeast-2'
    
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {}
        print("Payload: ", payload)
            
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
        )
        print("Invoked function %s.", function_name)
        print("Response: ", response)
    except ClientError:
        print("Couldn't invoke function %s.", function_name)
        raise
    
    payload = response['Payload']
    print('payload: ', payload)
    body = json.load(payload)['body']
    print('body: ', body)
    jsonBody = json.loads(body) 
    print('jsonBody: ', jsonBody)    
    timestr = jsonBody['timestr']
    print('timestr: ', timestr)
    
    result = []
    result.append(timestr)
    
    return result

@tool
def get_weather_info(city: str) -> str:
    """
    Search weather information by city name and then return weather statement.
    city: the english name of city to search
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
                
    if isKorean(city):
        place = traslation_to_english(chat, city)
        print('city (translated): ', place)
    else:
        place = city
    print('place: ', place)
    
    apiKey = api_key
    lang = 'en' 
    units = 'metric' 
    api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
    # print('api: ', api)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
            
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
            #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        # raise Exception ("Not able to request to LLM")    
    
    print('weather_str: ', weather_str)                            
    return weather_str

def get_react_prompt_template(mode: str): # (hwchase17/react) https://smith.langchain.com/hub/hwchase17/react
    # Get the react prompt template
    
    if mode=='eng':
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
    else: 
        return PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

Use the following format:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action으로서 [{tool_names}]중 하나를 선택합니다.
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 3번 반복 할 수 있습니다. 반복이 끝날때까지 정답을 찾지 못하면 마지막 result로 답변합니다.)
... (반복이 끝날때까지 적절한 답변을 얻지 못하면, 마지막 결과를 Final Answer를 전달합니다. )
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
        
# define tools
tools = [get_current_time, get_product_list, get_weather_info]        

def run_agent_react(connectionId, requestId, chat, query):
    prompt_template = get_react_prompt_template(agentLangMode)
    print('prompt_template: ', prompt_template)
    
    #from langchain import hub
    #prompt_template = hub.pull("hwchase17/react")
    #print('prompt_template: ', prompt_template)
    
     # create agent
    isTyping(connectionId, requestId)
    agent = create_react_agent(chat, tools, prompt_template)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations = 5
    )
    
    # run agent
    response = agent_executor.invoke({"input": query})
    print('response: ', response)

    # streaming    
    msg = readStreamMsg(connectionId, requestId, response['output'])

    msg = response['output']
    print('msg: ', msg)
            
    return msg

def run_agent_react_chat(connectionId, requestId, chat, query):
    # revise question
    revised_question = revise_question(connectionId, requestId, chat, query)     
    print('revised_question: ', revised_question)  
        
    # get template based on react 
    prompt_template = get_react_prompt_template(agentLangMode)
    print('prompt_template: ', prompt_template)
    
    # create agent
    isTyping(connectionId, requestId)
    agent = create_react_agent(chat, tools, prompt_template)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # run agent
    response = agent_executor.invoke({"input": revised_question})
    print('response: ', response)
    
    # streaming
    msg = readStreamMsg(connectionId, requestId, response['output'])

    msg = response['output']
    print('msg: ', msg)
            
    return msg

def run_agent_tool_calling(connectionId, requestId, chat, query):
    # toolList = "get_current_time, get_product_list, get_weather_info"
    toolList = ", ".join((t.name for t in tools))
    
    system = f"You are a helpful assistant. Make sure to use the {toolList} tools for information."
    # system = f"You are a helpful assistant. Make sure to use the get_current_time tool for information."
    #system = f"다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다. 답변에 필요한 정보는 다움의 tools를 이용해 수집하세요. Tools: {toolList}"
            
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            # ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), 
        ]
    )
    print('prompt: ', prompt)
    
     # create agent
    agent = create_tool_calling_agent(chat, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # run agent
    isTyping(connectionId, requestId)
    response = agent_executor.invoke({"input": query})
    print('response: ', response)

    # streaming        
    readStreamMsgForAgent(connectionId, requestId, response['output'])

    msg = response['output']    
    output = removeFunctionXML(msg)
    # print('output: ', output)
            
    return output

def run_agent_tool_calling_chat(connectionId, requestId, chat, query):
    toolList = "get_current_time, get_product_list, get_weather_info"
    # system = f"You are a helpful assistant. Make sure to use the {toolList} tools for information."
    system = f"다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다. 답변에 필요한 정보는 다움의 tools를 이용해 수집하세요. Tools: {toolList}"
            
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
     # create agent
    agent = create_tool_calling_agent(chat, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # run agent
    history = memory_chain.load_memory_variables({})["chat_history"]
    
    isTyping(connectionId, requestId)
    response = agent_executor.invoke({
        "input": query,
        "chat_history": history
    })
    print('response: ', response)

    # streaming        
    readStreamMsgForAgent(connectionId, requestId, response['output'])

    msg = response['output']    
    output = removeFunctionXML(msg)
    # print('output: ', output)
            
    return output

def traslation_to_english(chat, text):
    input_language = "Korean"
    output_language = "English"
        
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def revise_question(connectionId, requestId, chat, query):    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
        
    if isKorean(query)==True :      
        system = (
            ""
        )  
        human = """이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다. 결과는 <result> tag를 붙여주세요.
        
        <question>            
        {question}
        </question>"""
        
    else: 
        system = (
            ""
        )
        human = """Rephrase the follow up <question> to be a standalone question. Put it in <result> tags.
        <question>            
        {question}
        </question>"""
            
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        print('revised_question: ', revised_question)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
            
        sendDebugMessage(connectionId, requestId, f"새로운 질문: {revised_question}\n * 대화이력({str(history_length)}자, {token_counter_history} Tokens)을 활용하였습니다.")
            
    return revised_question    
    # return revised_question.replace("\n"," ")

def isTyping(connectionId, requestId):    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)

def removeFunctionXML(msg):
    #print('msg: ', msg)
    
    while(1):
        start_index = msg.find('<function_calls>')
        end_index = msg.find('</function_calls>')
        length = 18
        
        if start_index == -1:
            start_index = msg.find('<invoke>')
            end_index = msg.find('</invoke>')
            length = 10
        
        output = ""
        if start_index>=0:
            # print('start_index: ', start_index)
            # print('msg: ', msg)
            
            if start_index>=1:
                output = msg[:start_index-1]
                
                if output == "\n" or output == "\n\n":
                    output = ""
            
            if end_index >= 1:
                # print('end_index: ', end_index)
                output = output + msg[end_index+length:]
                            
            msg = output
        else:
            output = msg
            break

    return output

def readStreamMsgForAgent(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            output = removeFunctionXML(msg)
            # print('output: ', output)
            
            if len(output)>0 and output[0]!='<':
                result = {
                    'request_id': requestId,
                    'msg': output,
                    'status': 'proceeding'
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)
            
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)
    
def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)    
        
def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return msg

def getResponse(connectionId, jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
    
    global map_chain, memory_chain

    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')        
        memory_chain = map_chain[userId]
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
    
    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:            
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)                  
                elif convType == 'agent-react':
                    msg = run_agent_react(connectionId, requestId, chat, text)      
                elif convType == 'agent-react-chat':         
                    msg = run_agent_react_chat(connectionId, requestId, chat, text)
                elif convType == 'agent-toolcalling':
                    msg = run_agent_tool_calling(connectionId, requestId, chat, text)
                elif convType == 'agent-toolcalling-chat':         
                    msg = run_agent_tool_calling_chat(connectionId, requestId, chat, text)       
                                        
                elif convType == "translation":
                    msg = translate_text(chat, text) 
                elif convType == "grammar":
                    msg = check_grammer(chat, text)  
                else:
                    msg = general_conversation(connectionId, requestId, chat, text)  
                    
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                                        
        elif type == 'document':
            isTyping(connectionId, requestId)
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                docs = load_csv_document(path, doc_prefix, object)
                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                        
            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'uri': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            raise Exception ("Not able to write into dynamodb")               
        #print('resp, ', resp)

    return msg

def lambda_handler(event, context):
    # print('event: ', event)    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg = getResponse(connectionId, jsonBody)
                    # print('msg: ', msg)
                    
                    sendResultMessage(connectionId, requestId, msg)  
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }