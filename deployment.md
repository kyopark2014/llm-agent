# 인프라 설치하기

## Bedrock 사용 권한 설정하기

LLM으로 Anthropic의 Claude3을 사용하기 위하여, Amazon Bedrock의 Virginia(us-east-1) 리전을 사용합니다. [Model access - Virginia](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess)에 접속해서 [Edit]를 선택하여 모든 모델을 사용할 수 있도록 설정합니다. 

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/d0367605-0f23-47cb-a9f8-1c38bc1b0d0c)


## CDK를 이용한 인프라 설치하기

여기서는 [AWS Cloud9](https://aws.amazon.com/ko/cloud9/)에서 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 이용하여 인프라를 설치합니다. 또한 편의상 서울 리전을 통해 실습합니다.

1) [Cloud9 Console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/create)에 접속하여 [Create environment]-[Name]에서 “chatbot”으로 이름을 입력하고, EC2 instance는 “m5.large”를 선택합니다. 나머지는 기본값을 유지하고, 하단으로 스크롤하여 [Create]를 선택합니다.

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/c85c2ef5-4f96-4528-b5d4-ab9d3e52324e)

2) [Environment](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에서 “chatbot”를 [Open]한 후에 아래와 같이 터미널을 실행합니다.

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/fcf24f93-9ab3-4905-be8d-8146c7371951)

3) EBS 크기 변경

아래와 같이 스크립트를 다운로드 합니다. 

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

이후 아래 명령어로 용량을 80G로 변경합니다.
```text
chmod a+rx resize.sh && ./resize.sh 80
```


4) 소스를 다운로드합니다.

```java
git clone https://github.com/kyopark2014/llm-agent
```

5) cdk 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```java
cd llm-agent/cdk-llm-agent/ && npm install
```

7) CDK 사용을 위해 Boostraping을 수행합니다.

아래 명령어로 Account ID를 확인합니다.

```java
aws sts get-caller-identity --query Account --output text
```

아래와 같이 bootstrap을 수행합니다. 여기서 "account-id"는 상기 명령어로 확인한 12자리의 Account ID입니다. bootstrap 1회만 수행하면 되므로, 기존에 cdk를 사용하고 있었다면 bootstrap은 건너뛰어도 됩니다.

```java
cdk bootstrap aws://[account-id]/ap-northeast-2
```

8) 아래 명령어로 인프라를 설치합니다.

```java
cdk deploy --all
```

인프라가 설치가 되면 아래와 같은 Output을 확인할 수 있습니다. 

![noname](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/5c876093-3faa-4faf-a7d2-ed9659a75473)

9) Output의 WebUrlforstreamchatbot의 URL로 접속합니다. 

