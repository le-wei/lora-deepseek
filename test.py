from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
import os
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM, GenerationConfig
def original_model_reasoning(model_path: str, prompt: str, max_new_tokens=2048):
    """
    单论对话的回复
    :param model_path: 模型下载地址
    :param prompt: 需要询问的问题
    :return: 回复的话
    """

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 device_map="auto")

    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = [
        {"role": "user",
         "content": prompt}
    ]

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=max_new_tokens)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print(f"result: {result}")
    return result

original_model_reasoning(
    model_path='/data_code/code/deepseek-llm-7b-chat', #/data_code/code/deepseek-llm-7b-chat
    prompt='现在有点咽喉痛，是什么原因导致'
)