from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
import os
from modelscope import AutoTokenizer
import shutil

# 保证原始模型的各个文件不遗漏保存到merge_path中
def copy_files_not_in_B(A_path, B_path):
    """
    Copies files from directory A to directory B if they exist in A but not in B.

    :param A_path: Path to the source directory (A).
    :param B_path: Path to the destination directory (B).
    """
    # 确保源目录 A 存在，目标目录 B 存在，如果不存在则创建
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    # 获取源目录 A 中的所有文件和文件夹
    files_in_A = os.listdir(A_path)
    # 过滤掉权重文件（如 .bin 和 .safetensors 文件），只保留其他文件
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])
    # 获取目标目录 B 中的所有文件和文件夹
    files_in_B = set(os.listdir(B_path))

    # 找出源目录 A 中存在但目标目录 B 中不存在的文件或文件夹
    files_to_copy = files_in_A - files_in_B

    # 遍历需要复制的文件或文件夹
    for file in files_to_copy:
        # 构造源文件路径和目标文件路径
        src_path = os.path.join(A_path, file)
        dst_path = os.path.join(B_path, file)

        if os.path.isdir(src_path):
            # 如果是目录，则递归复制整个目录及其内容
            shutil.copytree(src_path, dst_path)
        else:
            # 如果是文件，则复制文件
            shutil.copy2(src_path, dst_path)

def merge_lora_to_base_model():
    """
    将 LoRA 微调的权重合并到基础模型中，并保存合并后的模型。

    功能：
        1. 加载基础模型和 LoRA 微调的权重。
        2. 将 LoRA 的权重合并到基础模型中。
        3. 保存合并后的模型到指定路径，同时保留基础模型的其他文件。

    参数:
        无直接参数，路径和配置在函数内部定义。
    """
    # 定义基础模型路径、LoRA 微调权重路径和保存合并模型的路径
    model_name_or_path = '/data_code/code/deepseek-llm-7b-chat'  # 原模型地址
    adapter_name_or_path = '/data_code/code/GRPO/data/LORA_DEEPSEEK'  # 微调后模型的保存地址
    save_path = '/data_code/code/GRPO/data/LORA_DEEPSEEK/moss-10000-4096-16-32-epoch-2-merge-model'

    # 如果文件夹不存在，就创建
    # 如果保存路径不存在，则创建该目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 从基础模型路径加载分词器，设置 `trust_remote_code=True` 以支持自定义模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,)
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,                     # 基础模型路径
        trust_remote_code=True,                 # 支持自定义模型
        low_cpu_mem_usage=True,                 # 减少 CPU 内存使用
        torch_dtype=torch.float16,              # 数据类型设置为 float16，以节省内存        
        device_map="auto"                       # 自动分配设备（如 GPU）
    )
    # 加载保存的 Adapter
    # 使用 PEFT 的 `PeftModel` 加载微调后的权重，并将其应用到基础模型
    model = PeftModel.from_pretrained(model, adapter_name_or_path,  # LoRA 微调权重的路径
                                      device_map=None,              # 不重新分配设备
                                      trust_remote_code=True        # 支持自定义模型
                                      )
    # 将 Adapter 合并到基础模型中
    # `merge_and_unload` 是 PEFT 提供的方法，用于将 LoRA 的权重合并到基础模型，并卸载 LoRA 适配器
    merged_model = model.merge_and_unload()  # PEFT 的方法将 Adapter 权重合并到基础模型
    # 保存合并后的模型
    # 保存分词器到指定路径
    tokenizer.save_pretrained(save_path)
    # 保存合并后的模型权重到指定路径
    merged_model.save_pretrained(save_path, safe_serialization=False)
    # 确保基础模型的其他文件（如配置文件）也被复制到保存路径中
    copy_files_not_in_B(model_name_or_path, save_path)
    print(f"合并后的模型已保存至: {save_path}")

if __name__ == '__main__':
    merge_lora_to_base_model()
