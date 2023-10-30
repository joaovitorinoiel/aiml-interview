# General
import os
import json
import boto3

# LLM
from langchain.llms import Bedrock

BUCKET = os.environ["BUCKET"]
s3 = boto3.client("s3")

LLM_MODEL_ID = "anthropic.claude-v2"

metrics = {
    "transcription": "",
    "feedback": "",
}

bedrock = boto3.client(
    "bedrock-runtime",
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    region_name="us-east-1",
)

# Ask LLM
llm_mentor_response = Bedrock(
    model_id=LLM_MODEL_ID,
    model_kwargs={
        "max_tokens_to_sample": 1000,
        "temperature": 0.1,
        "top_p": 0.3,
    },
    client=bedrock,
)


def bedrock_feedback(text):
    prompt_template_bedrock = """
    Human: De um feedback em português para a apresentação que o mentorado fez simulando uma entrevista destaque os pontos de melhoria e de sugestões. A apresentação do cadidato é a seguinte:
    
    <apresentação>'{text}'</apresentação>
    
    De acordo com a apresentação do aluno e não de forma genérica, o feedback deve conter:
    - Resumo da avaliação;
    - Palavras para serem evitadas;
    - 3 pontos positivos;
    - 3 pontos negativos;

    O feedback deve seguir o seguinte formato:
    Resumo:
    <resumo></resumo>
    Palavras para serem evitadas:
    <palavras></palavras>
    Pontos positivos e negativos:
    <positivos></positivos>
    <negativos></negativos>
    
    Assistant:
    """
    chain = llm_mentor_response(prompt_template_bedrock)
    return str(chain)


def lambda_handler(event, context):
    transcription_file = event["TranscriptionJob"]["Transcript"][
        "TranscriptFileUri"
    ]

    key = os.path.splitext(os.path.basename(transcription_file))
    s3.download_file(
        BUCKET, "transcription/" + key[0] + key[1], "/tmp/transcription.json"
    )
    file = open("/tmp/transcription.json")

    data = json.load(file)
    text = data["results"]["transcripts"][0]["transcript"]
    metrics["transcription"] = text
    metrics["feedback_bedrock"] = bedrock_feedback(text).replace('"', "`")

    return {
        "statusCode": 200,
        "body": {"metrics": str(metrics)},
    }
