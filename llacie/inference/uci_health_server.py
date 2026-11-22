import os
import re

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError
from pathlib import Path

DEFAULT_MODEL = 'openai/gpt-oss-120b'


def api_model_name():
    """The model requested from the API server: MODEL_NAME from the environment, then .env,
    then .env.example (same precedence as llacie's Config), falling back to DEFAULT_MODEL."""
    settings = {**dotenv_values(".env.example"), **dotenv_values(".env"), **os.environ}
    return (settings.get('MODEL_NAME') or '').strip() or DEFAULT_MODEL


def api_model_label():
    """api_model_name() made safe for a strategy name, e.g. openai/gpt-oss-120b ->
    openai_gpt_oss_120b. Strategy names use dots as separators, so dots are replaced too."""
    return re.sub(r'[^a-z0-9]+', '_', api_model_name().lower()).strip('_')


class UCI_HEALTH:

    def __init__(self):
        self.SYSTEM_PROMPT = """ 
                You are a clinical researcher that reads medical charts and answers questions about them. 
                Use only the information in the text provided to answer the question.
                If a patient denies something, do not include it in your answer.
                After you provide an answer, you immediately stop talking
        """
            
        self.config = {}
        
        self.DEFAULT_MODEL = DEFAULT_MODEL
    
    def load_config(self):
        """Loads config file from the environment"""
        self.config.update({**dotenv_values(".env")})


    def create_client(self,**kwargs):
        """ Connects to the UCI Server using base_url and API_key"""
        client = OpenAI(
            api_key=self.config['API_KEY'],
            base_url=self.config['BASE_URL'],
            **kwargs)
        return client

    def create_query(self,messages:str,model=None, reasoning_effort='high',temperature=0.5,seed=0,client=None)->tuple:
        """ Sends query to backend via the OpenAI Responses API, returns tuple of messages,
        reasoning.

        `reasoning_effort` (low/medium/high) is sent as `reasoning.effort`; None, '' or 'none'
        leaves it out. The Responses API accepts it for GLM too, where chat completions'
        `reasoning_effort` was rejected. Reasoning comes back as raw reasoning text
        (gpt-oss) or as a reasoning summary (GLM); either is returned. """

        if client is None:
            client = self.create_client()

        if type(messages) is not str:
            messages = self.USER_PROMPT

        reasoning_params = {"summary": "auto"}
        if reasoning_effort is not None and str(reasoning_effort).strip().lower() not in ('', 'none'):
            reasoning_params["effort"] = str(reasoning_effort).strip()

        stream = client.responses.create(
            model=api_model_name() if model is None else model,
            instructions=self.SYSTEM_PROMPT,
            input=messages,
            reasoning=reasoning_params,
            temperature=temperature,
            # The Responses API has no seed parameter; pass it through for servers that use it
            extra_body={"seed": seed} if seed is not None else None,
            stream=True,
        )

        response_chunks = []
        reasoning_chunks = []
        for event in stream:
            if event.type == "response.output_text.delta":
                response_chunks.append(event.delta)
            elif event.type in ("response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta"):
                reasoning_chunks.append(event.delta)
            elif event.type in ("response.failed", "error"):
                raise OpenAIError(f"Response failed: {getattr(event, 'response', event)}")

        response = ''.join(response_chunks)
        reasoning = ''.join(reasoning_chunks)
        return response, reasoning


def read_query(query):
    print(query)


def main(message):
    server = UCI_HEALTH()
    server.load_config()
    messages = message
    query,reasoning = server.create_query(messages)
    i = 0
    with open('reasoning.txt','w') as f:
        f.write(f"Reasoning Logic of {i}")
        f.write(reasoning)
        i+=1
    read_query(query)


if __name__ == '__main__':
    raise NotImplementedError