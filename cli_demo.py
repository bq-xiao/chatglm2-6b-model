import os
import platform

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/chatglm2-6b-model/model", trust_remote_code=True)
model = AutoModel.from_pretrained("/kaggle/working/chatglm2-6b-model/model", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
past_key_values, history = None, []
print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
while True:
    query = input("\n用户：")
    if query.strip() == "stop":
        break
    if query.strip() == "clear":
        past_key_values, history = None, []
        os.system(clear_command)
        print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        continue
    print("\nChatGLM：", end="")
    current_length = 0
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                past_key_values=past_key_values,
                                                                return_past_key_values=True):
        if stop_stream:
            stop_stream = False
            break
        else:
            print(response[current_length:], end="", flush=True)
            current_length = len(response)
    print("")
