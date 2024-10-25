"""
openai, llava request&response
"""
import os
import json
import time
import base64
import requests
import jsonlines
from retrying import retry


# openai configuration

import openai


openai_api_key = ""  # openai api key

others_api_key = "xxxxxxxxxxxxxxxxxxxxxx"  # other LLM api key, e.g. zhipu GLM

openai_organization = None

openai_proxy = {
    'http':"http://127.0.0.1:7890",
    'https':"http://127.0.0.1:7890",
    }

openai_model = "gpt-3.5-turbo"

# openai modules

def get_model():
    global openai_model
    return openai_model

def configure_openai(model = "gpt-3.5-turbo", is_openai=True, ip_port = None):
    """
    function: configure openai's api_key and proxy
            or configure other model using openai's format
    """
    global openai_model
    if is_openai:
        openai.api_key = openai_api_key
        openai.organization = openai_organization
        openai.proxy = openai_proxy
        openai_model = model
    else:
        openai.api_key = others_api_key
        openai.api_base = ip_port
        openai_model = model
    return 0

def chatgpt_request(messages, model_name=None, temperature=0.9, max_tokens=4095):
    """ 
    function: make chatgpt request
    input   : messages = [{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}]
    output  : response, tokens_count
    """
    global openai_model
    try:
        if model_name is None:      
            model = openai_model
        else:
            model = model_name
        completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        return completion["choices"][0]["message"]["content"], completion['usage']['total_tokens']
    except Exception as e:
        print("ChatGPT ERROR:", e)
        return "ChatGPT ERROR", 0


def record_id(seed):
    """
    function: generate random id
    """
    from faker import Faker
    fake = Faker()
    Faker.seed(seed)
    return str(fake.uuid4()).split("-")[-1]

def write_gpt_label_data(dialogs, output_path="log/defaultlog.jsonl", task_name="llmsearch", model_name=None):
    """
    function: get gpt response and save the log
    """
    if model_name is None:      
        model = openai_model
    else:
        model = model_name
    with jsonlines.open(output_path, "a") as wid:
        content, tokens = chatgpt_request(dialogs, model)
        dialogs.append({"role":"assistant", "content":content})
        info = {}
        info["task"] = task_name
        info["id"] = record_id(int(time.time()))
        info["time"] = time.asctime(time.localtime(time.time()))
        info["diag"] = dialogs
        info["model"] = model
        wid.write(info)
    
    return content, tokens

def write_gpt_data(dialogs, response, output_path="log/defaultlog.jsonl", task_name="llmtest", model_name=None):
    """
    function: save the log
    """
    if model_name is None:      
        model = openai_model
    else:
        model = model_name

    with jsonlines.open(output_path, "a") as wid:
        dialogs.append({"role":"assistant", "content":response})
        info = {}
        info["task"] = task_name
        info["id"] = record_id(int(time.time()))
        info["time"] = time.asctime(time.localtime(time.time()))
        info["diag"] = dialogs
        info["model"] = model
        wid.write(info)


# llava modules

llava_response_cache = "dataset/llava_response_local/llava_response_local.json" 

def get_model():
    global llava_response_cache
    return llava_response_cache

def configure_llava(local_cache):
    global llava_response_cache
    llava_response_cache = local_cache

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def send_llava_requests(url, data, port):
    # send POST request to Flask 
    response = requests.post(url, json=data, timeout=5)  
    return response

def llava_predict(image, text, port=5000, path_only=False) -> str:
    """
    function: make llava predict using the flask app
    """

    # read image file and encode it to Base64 
    if not path_only:
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        encoded_image = image

    data = {
        'image': encoded_image,
        'text': text,
        'path_only': path_only
    }

    try:
        response = send_llava_requests(f'http://localhost:{port}/predict', data, port)  #  Flask run in localhost: port
        # get response
        result = response.json()
        return result['result']
    except Exception as e:
        print(f"Failed to send request: {e}")
        raise RuntimeError(f"Llava predict response error by {e}")


def llava_predict_local(image, text, local_response=None, port=5000, path_only=False) -> str:
    global llava_response_cache
    if local_response is None:
        cache_file = llava_response_cache
    else:
        cache_file = local_response
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            llava_response = json.load(f)
    else:
        llava_response = {}

    updated = False

    if image in llava_response.keys():
        if text in llava_response[image].keys():
            result = llava_response[image][text]
        else:
            result = llava_predict(image, text, port, path_only)
            llava_response[image][text] = result
            updated = True
    else:
        llava_response[image] = {}
        result = llava_predict(image, text, port, path_only)
        llava_response[image][text] = result
        updated = True
    
    if updated:
        with open(cache_file, "w") as f:
            json.dump(llava_response, f)
    return result