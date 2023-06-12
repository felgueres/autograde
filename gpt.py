import os
import openai
import backoff
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
else:
    print('OpenAI key not set')

completion_tokens = prompt_tokens = 0

class OpenAIError(Exception):
    pass 

@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model='gpt-3.5-turbo', temperature=0.5, max_tokens=1000, n=1, stop=None):
    messages = [{'role': 'user', 'content': prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-3.5-turbo", temperature=0.5, max_tokens=1000, n=1, stop=None):
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stop=stop, n=cnt)
        outputs.extend([choice['message']['content'] for choice in res['choices']])
    return outputs