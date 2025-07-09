import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "./Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("Qwen3-0.6B 对话系统已启动（输入 'exit' 退出）\n")

while True:
    # 用户输入
    user_input = input("用户: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    # 构建对话模板
    messages = [{"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回复
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024  # 调整为更合理的值
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    end_time = time.time()
    res_time = end_time-start_time
    
    # 输出结果
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"\nAI: {response}")
    print(f"生成统计: {len(output_ids)} tokens | 耗时: {res_time:.2f}s")
    print(f"每个Tokens耗时: {res_time:.2f / len(output_ids) }s\n")  # 计算生成速度

print("对话已结束")