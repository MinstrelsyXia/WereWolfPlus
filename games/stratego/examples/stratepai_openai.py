# This code is for v1 of the openai package: pypi.org/project/openai
# from openai import OpenAI
# from my_env import API_KEY
# client = OpenAI(api_key=API_KEY)
import openai
from agent_manager.prompts.stratego_prompt import StategoPrompt
def get_openAI_move(game_state_report: str) -> str:
    # print("game_state_report:====")
    # print(game_state_report)
    model_dict={
        "model":"deepseek-chat",
        "temperature":0.7,
        "max_tokens":1280,
        "api_key":'sk-c7743bb83dc54d078af3731f4d7885e1',
        "api_base":'https://api.deepseek.com',

    }

    messages = [
        {
            "role": "system",
            "content": StategoPrompt.sys_prompt()
        },
        {
            "role": "user",
            "content": game_state_report,
        }
    ],
    openai.api_key = model_dict['api_key']
    openai.api_base = model_dict['api_base']
    output = openai.ChatCompletion.create(
        model=model_dict['model'],
        messages=messages[0],
        temperature=model_dict['temperature']
    )



    # print("~~~~~~~~~")
    # print(answer)
    # print("~~~~~~~~~")
    # return answer

# print(get_openAI_move(''))
