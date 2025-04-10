import argparse
from os.path import join

import pandas as pd
from datasets import Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback
import bitsandbytes as bnb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import json
from datasets import load_dataset

# 配置参数
def configuration_parameter():
    """
    配置并解析命令行参数，用于DeepSeek模型的LoRA微调。

    返回:
        argparse.Namespace: 包含所有解析后的命令行参数的命名空间对象。
    """
    # 创建一个参数解析器对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for deepseek model")

    # 模型路径相关参数
    # 指定本地下载的模型目录路径
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-llm-7b-chat",
                        help="Path to the model directory downloaded locally")
    # 指定保存微调后模型和检查点的目录
    parser.add_argument("--output_dir", type=str,
                        default="/data_code/code/GRPO/data/LORA_DEEPSEEK",
                        help="Directory to save the fine-tuned model and checkpoints")

    # 数据集路径
    # 指定JSONL格式的训练数据文件路径
    parser.add_argument("--train_file", type=str, default="/data_code/code/GRPO/data/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json",
                        help="Path to the training data file in JSONL format")

    # 训练超参数
    # 指定训练的轮数
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    # 指定每个设备在训练期间的批次大小
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,#4,
                        help="Batch size per device during training")
    # 指定在执行反向传播/更新步骤之前要累积的更新步数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,#16,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    # 指定优化器的学习率
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer")
    # 指定输入的最大序列长度
    parser.add_argument("--max_seq_length", type=int, default=1024,#2048,
                        help="Maximum sequence length for the input")
    # 指定记录指标的步数间隔
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of steps between logging metrics")
    # 指定保存检查点的步数间隔
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Number of steps between saving checkpoints")
    # 指定要保留的最大检查点数量
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")
    # 指定学习率调度器的类型
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                        help="Type of learning rate scheduler")
    # 指定学习率调度器的热身步数
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduler")

    # LoRA 特定参数
    # 指定LoRA矩阵的秩
    parser.add_argument("--lora_rank", type=int, default=64, #64,
                        help="Rank of LoRA matrices")
    # 指定LoRA的Alpha参数
    parser.add_argument("--lora_alpha", type=int, default= 16 ,#16,
                        help="Alpha parameter for LoRA")
    # 指定LoRA的丢弃率
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")

    # 分布式训练参数
    # 指定分布式训练的本地排名
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Local rank for distributed training")
    # 指定是否启用分布式训练
    parser.add_argument("--distributed", type=bool, default=False, help="Enable distributed training")

    # 额外优化和硬件相关参数
    # 指定是否启用梯度检查点以节省内存
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing to save memory")
    # 指定训练期间使用的优化器
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")
    # 指定训练模式，支持 'lora' 和 'qlora'
    """
    具体作用：
    lora 模式：
        使用 LoRA（Low-Rank Adaptation）技术对模型进行微调。
        LoRA 通过在模型的全连接层中插入低秩矩阵来减少训练参数的数量，从而降低显存占用和计算成本。
    qlora 模式：
        使用 QLoRA（Quantized LoRA）技术进行微调。
        QLoRA 是 LoRA 的量化版本，通常会将模型的权重量化为 4 位（4-bit），进一步减少显存占用，同时保持较高的性能
    """
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    # 指定随机种子以确保结果可复现
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # 指定是否使用混合精度（FP16）训练
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Use mixed precision (FP16) training")
    # 指定用于记录日志的报告工具（如tensorboard）
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    # 指定数据加载的工作线程数
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    # 指定保存检查点的策略（'steps'或'epoch'）
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Strategy for saving checkpoints ('steps', 'epoch')")
    # 指定优化器的权重衰减
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay for the optimizer")
    # 指定梯度裁剪的最大梯度范数
    parser.add_argument("--max_grad_norm", type=float, default=1,
                        help="Maximum gradient norm for clipping")
    # 指定是否从数据集中移除未使用的列
    parser.add_argument("--remove_unused_columns", type=bool, default=True,
                        help="Remove unused columns from the dataset")
    # 解析命令行参数
    args = parser.parse_args()
    return args


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter

    参数:
        model (torch.nn.Module): 要查找全连接层的模型。
        train_mode (str): 训练模式，支持 'lora' 或 'qlora'。

    返回:
        list: 包含所有需要添加 LoRA 适配器的全连接层名称的列表。
    """
    # 确保训练模式为 'lora' 或 'qlora'
    assert train_mode in ['lora', 'qlora']
    # 根据训练模式选择线性层的类
    # 如果是 'qlora' 模式，选择 4-bit 量化的线性层 (bnb.nn.Linear4bit)
    # 如果是 'lora' 模式，选择标准的线性层 (torch.nn.Linear) 
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    # 用于存储需要添加 LoRA 适配器的模块名称
    lora_module_names = set()
    # 遍历模型中的所有子模块
    for name, module in model.named_modules():
        # 如果模块是指定的线性层类型 (cls)，则记录其名
        if isinstance(module, cls):
            # 将模块名称分割为层级结构
            names = name.split('.')
            # 如果名称只有一级，直接添加；否则添加最后一级名称
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # 如果 'lm_head' 模块在列表中，移除它
    # 通常 'lm_head' 是模型的输出层，不需要添加 LoRA 适配器 ，因为输出层通常不需要LoRA适配器
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    # 将模块名称集合转换为列表
    lora_module_names = list(lora_module_names)
    # 记录找到的模块名称，用于调试或日志记录
    logger.info(f'LoRA target module names: {lora_module_names}')
    # 返回需要添加 LoRA 适配器的模块名称列表
    return lora_module_names


def setup_distributed(args):
    """
    初始化分布式训练环境。

    参数:
        args (argparse.Namespace): 包含命令行参数的命名空间对象，其中包括分布式训练相关的参数。
    """
    # 检查是否启用了分布式训练
    if args.distributed:
        # 如果启用了分布式训练，但 local_rank 参数未正确设置，则抛出错误
        if args.local_rank == -1:
            raise ValueError("未正确初始化 local_rank，请确保通过分布式启动脚本传递参数，例如 torchrun。")

        # 初始化分布式进程组
        # backend="nccl" 表示使用 NVIDIA 的 NCCL 库进行 GPU 间通信（适用于多 GPU 环境）
        dist.init_process_group(backend="nccl")
        # 设置当前进程使用的 GPU 设备
        # local_rank 表示当前进程的 GPU 索引
        torch.cuda.set_device(args.local_rank)
        # 打印分布式训练的初始化信息
        print(f"分布式训练已启用，Local rank: {args.local_rank}")
    else:
        print("未启用分布式训练，单线程模式。")


# 加载模型
def load_model(args, train_dataset, data_collator):
    # 初始化分布式环境
    """
    加载并配置模型，同时初始化分布式训练环境和 LoRA 参数。

    参数:
        args (argparse.Namespace): 包含命令行参数的命名空间对象。
        train_dataset (Dataset): 训练数据集。
        data_collator (DataCollator): 数据整理器，用于批量处理数据。

    返回:
        Trainer: 配置好的 Trainer 对象，用于训练模型。
    """
    # 初始化分布式训练环境
    setup_distributed(args)
    # 自动分配设备
    # 加载模型
    # 配置模型加载参数
    model_kwargs = {
        "trust_remote_code": True,      # 允许加载远程代码（适用于自定义模型）
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,  # 根据参数选择 FP16 或 BF16 精度
        "use_cache": False if args.gradient_checkpointing else True, # 如果启用梯度检查点，则禁用缓存
        # "device_map": "auto" if not args.distributed else None,
    }
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    # 用于确保模型的词嵌入层参与训练
    # 确保模型的词嵌入层参与训练
    model.enable_input_require_grads()
    # 将模型移动到正确设备
    # 如果启用了分布式训练，将模型移动到指定的 GPU 并包装为 DDP（分布式数据并行）
    if args.distributed:
        model.to(args.local_rank)   # 将模型移动到指定的 GPU
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # 哪些模块需要注入Lora参数
    # 查找需要注入 LoRA 参数的模块
    target_modules = find_all_linear_names(model.module if isinstance(model, DDP) else model, # 如果是 DDP 模型，获取其原始模型
                                           args.train_mode                                  # 根据训练模式（lora 或 qlora）查找模块
                                           )
    # lora参数设置
    
    config = LoraConfig(
        r=args.lora_rank,                   # LoRA 矩阵的秩
        lora_alpha=args.lora_alpha,         # LoRA 矩阵的 Alpha 参数
        lora_dropout=args.lora_dropout,     # LoRA 矩阵的丢弃率
        bias="none",                        # 不使用偏置项
        target_modules=target_modules,      # 需要注入 LoRA 参数的模块
        task_type=TaskType.CAUSAL_LM,       # 任务类型，适用于因果语言模型
        inference_mode=False                # 启用训练模式

    )
    use_bfloat16 = torch.cuda.is_bf16_supported()  # 检查设备是否支持 bf16
    # 配置训练参数
    train_args = TrainingArguments(
        output_dir=args.output_dir,     # 模型保存路径
        per_device_train_batch_size=args.per_device_train_batch_size,   # 每个设备的批量大小
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # 梯度累积步数
        logging_steps=args.logging_steps,   # 记录日志的步数
        num_train_epochs=args.num_train_epochs,   # 训练的总轮数
        save_steps=args.save_steps,   # 保存模型的步数
        learning_rate=args.learning_rate,   # 学习率
        save_on_each_node=True,   # 每个节点保存模型
        gradient_checkpointing=args.gradient_checkpointing,   # 启用梯度检查点以节省显存
        report_to=args.report_to,   # 日志报告工具（如 tensorboard）
        seed=args.seed,           # 随机种子
        optim=args.optim,         # 优化器类型
        local_rank=args.local_rank if args.distributed else -1, # 分布式训练的本地 rank
        ddp_find_unused_parameters=False,  # 禁用未使用参数检查以优化分布式训练
        fp16=args.fp16, # 是否启用 FP16 精度
        bf16=not args.fp16 and use_bfloat16,    # 如果未启用 FP16 且设备支持 BF16，则启用 BF16
        remove_unused_columns=False # 不移除未使用的列
    )
    # 应用 PEFT 配置到模型
    model = get_peft_model(model.module if isinstance(model, DDP) else model, config)  # 确保传递的是原始模型
    print("model:", model)
    model.print_trainable_parameters()  # 打印可训练参数的数量

    # 配置 SwanLab 平台的回调，用于实验管理和日志记录
    swanlab_config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dataset": "medical-o1-reasoning-SFT" # 数据集名称

    }
    swanlab_callback = SwanLabCallback(
        project="deepseek-finetune",                # 项目名称
        experiment_name="deepseek-llm-7b-chat-lora",    # 实验名称
        description="使用 medical-o1-reasoning-SFT 采用LoRA 微调 DeepSeek 模型",  # 实验描述
        workspace=None, # 工作空间
        config=swanlab_config,  # 配置参数
    )
    # 创建 Trainer 对象
    trainer = Trainer(
        model=model,                # 模型
        args=train_args,            # 训练参数
        train_dataset=train_dataset,    # 训练数据集
        data_collator=data_collator,    # 数据整理器
        callbacks=[swanlab_callback],   # 回调函数
    )
    return trainer


# 处理数据
def process_data(data: dict, tokenizer, max_seq_length):
    """
    处理单条数据样本，将其转换为模型可接受的输入格式。

    参数:
        data (dict): 单条数据样本，包含问题、复杂推理和回答等字段。
        tokenizer (transformers.PreTrainedTokenizer): 用于将文本转换为模型输入的分词器。
        max_seq_length (int): 输入序列的最大长度，超过该长度时会进行截断。

    返回:
        dict: 包含模型输入所需的 input_ids、attention_mask 和 labels。
    """

    # 初始化存储模型输入的列表
    input_ids, attention_mask, labels = [], [], []

    # 获取用户问题和模型回答 
    human_text = data["Question"].strip()
    assistant_text = "\n### Thinking: "+ data["Complex_CoT"].strip()  + "\n### Answer:" + data["Response"].strip()  # 模型的复杂推理部分 + # 模型的最终回答
    # 构造输入文本，包含用户问题和模型回答的提示
    input_text = "Human:" + human_text + "\n\nnAssistant:"
    # 对输入文本进行分词
    input_tokenizer = tokenizer(
        input_text,
        add_special_tokens=False,   # 不添加特殊标记（如 [CLS], [SEP]）
        truncation=True,            # 启用截断，确保长度不超过模型支持的最大长度
        padding=False,              # 不进行填充
        return_tensors=None,        # 返回普通的 Python 列表，而不是张量
    )
    # 对输出文本（模型的推理和回答）进行分词
    output_tokenizer = tokenizer(
        assistant_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    # 构造 input_ids（输入 ID 列表）
    # 包括输入文本的 token IDs、输出文本的 token IDs，以及结束标记（eos_token_id）
    input_ids += (
            input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
    )
    # 构造 attention_mask（注意力掩码）
    # 输入和输出的 token 都设置为 1，表示需要关注
    attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
    # 构造 labels（标签）
    # 输入部分的标签设置为 -100（表示忽略），输出部分的标签为实际的 token IDs
                        # 忽略输入部分的标签                       # 输出部分的标签                 # 添加结束标记
    labels += ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
                )
    # 如果序列长度超过最大长度，则进行截断
    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    # 返回处理后的数据
    return {
        "input_ids": input_ids,             # 模型的输入 ID
        "attention_mask": attention_mask,   # 注意力掩码
        "labels": labels                    # 标签（用于计算损失）
    }



# 训练部分
def main():
    args = configuration_parameter()
    print("*****************加载分词器*************************")
    # 加载分词器
    model_path = args.model_name_or_path
    # 从指定的模型路径加载分词器，设置 `trust_remote_code=True` 以支持自定义模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print("*****************处理数据*************************")
    # 处理数据
    # 获得数据
    # 从 JSON 文件中加载数据集，读取前 1500 条数据作为训练集
    dataset = load_dataset("json", data_files=args.train_file, split="train[:1500]")
    print("数据集大小：", dataset)

    # 使用 `process_data` 函数对数据集进行预处理
    # `process_data` 会将每条数据转换为模型可接受的格式，包括 `input_ids`、`attention_mask` 和 `labels`
    train_dataset = dataset.map(process_data,
                                 fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length}, # 传递分词器和最大序列长度
                                 remove_columns=dataset.column_names)                                       # 移除原始数据中的列，只保留处理后的数据
    # `DataCollatorForSeq2Seq` 用于对批量数据进行动态填充（padding），并返回 PyTorch 张量
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,     # 使用加载的分词器
                                           padding=True,            # 启用动态填充
                                           return_tensors="pt"      # 返回 PyTorch 张量
                                           )
    print(train_dataset, data_collator)
    # 加载模型
    print("*****************训练*************************")
    # 调用 `load_model` 函数，加载预训练模型并配置 LoRA 参数和分布式训练环境
    trainer = load_model(args, train_dataset, data_collator)
    trainer.train()
    # 训练
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)


if __name__ == "__main__":
    main()

