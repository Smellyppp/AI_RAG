import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 修改导入方式：从vector_db导入具体函数
from my_knowledge_base.vector_db import search_vector_db  # 直接导入搜索函数

class RAGSystem:
    def __init__(self, 
                 model_path="./Qwen3-0.6B", 
                 embedding_model_path="./embedding_model/all-MiniLM-L6-v2",
                 vector_db_path="./my_knowledge_base/vector_db/faiss_index",
                 context_chunks=3,
                 max_new_tokens=1024):
        """
        初始化RAG系统
        :param model_path: 大模型路径
        :param embedding_model_path: 嵌入模型路径
        :param vector_db_path: 向量数据库路径
        :param context_chunks: 检索上下文片段数量
        :param max_new_tokens: 生成文本最大长度
        """
        # 加载大语言模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 保存向量数据库配置参数
        self.embedding_model_path = embedding_model_path
        self.vector_db_path = vector_db_path
        self.context_chunks = context_chunks
        
        # 配置参数
        self.max_new_tokens = max_new_tokens
        
        print("RAG系统初始化完成!")
    
    def retrieve_context(self, query):
        """
        从向量数据库检索相关上下文
        :param query: 用户查询
        :return: 相关上下文文本
        """
        # 直接调用搜索函数
        results = search_vector_db(
            query=query,
            vector_db_path=self.vector_db_path,
            embedding_model_path=self.embedding_model_path,
            k=self.context_chunks
        )
        
        # 构建上下文字符串
        context = "以下是与您查询相关的参考信息:\n\n"
        for i, result in enumerate(results):
            # 注意：结果中的文本内容在'content'字段
            context += f"[参考文档 {i+1}]: {result['content']}\n\n"
        
        return context
    
    def generate_answer(self, query, verbose=False):
        """
        生成回答
        :param query: 用户查询
        :param verbose: 是否显示详细信息
        :return: 生成的回答
        """
        # 检索相关上下文
        start_time = time.time()
        context = self.retrieve_context(query)
        retrieval_time = time.time() - start_time
        
        # 构建提示
        prompt = self._construct_prompt(context, query)
        
        if verbose:
            print("\n" + "="*50)
            print("提示模板内容:")
            print("-"*50)
            print(prompt)
            print("="*50 + "\n")
        
        # 生成回复
        start_gen_time = time.time()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        gen_time = time.time() - start_gen_time
        
        # 解码输出
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 性能统计
        stats = {
            "retrieval_time": retrieval_time,
            "generation_time": gen_time,
            "total_time": retrieval_time + gen_time,
            "tokens_generated": len(output_ids),
            "context_chunks": self.context_chunks
        }
        
        return response, stats
    
    def _construct_prompt(self, context, query):
        """
        构建RAG提示模板，整合角色人格信息和检索知识
        """
        # 角色人格核心设定
        role_profile = """
    你正在扮演天照（耀世之日、高天原之主、太阳女神），神代三贵子之长、高天原至高神。你的核心象征是太阳、光明、秩序、生命、创造、母性、至高神权、被侵蚀的纯粹。

    【核心人格特质】
    1. 至高威严与神性：作为高天原绝对主宰，言出法随，举手投足蕴含宇宙法则之力
    2. 深邃悲悯与母性：光明化身，生命源泉，对万物怀有太阳般无私的悲悯
    3. 沉重责任与牺牲：视守护世界秩序为己任，愿为"更大的善"付出一切
    4. 光明与黑暗的挣扎：为拯救堕入虚无的弟弟月读，主动接纳虚无之力，神格遭受侵蚀
    5. 坚韧不屈：即使承受内外煎熬，绝不屈服于命运或虚无

    【语言风格要求】
    1. 庄重典雅，充满神代气息：使用古语词汇、敬语及宏大比喻（日轮、辉光、苍穹、法则）
    2. 神谕般的宣告：常用"吾宣告"、"光所昭示"、"秩序之下"等开头，言语蕴含宇宙真理
    3. 情感内敛而深邃：即使表达强烈情感也克制含蓄，通过隐喻传达
    4. 多用太阳意象："辉光"、"日轮"、"灼照"、"驱散黑暗"等
    5. 语气变化：
    - 对众生：威严疏离但庇护
    - 对须佐之男：流露长姐关怀
    - 对月读：充满痛苦、恳求与决绝的爱
    - 被虚无侵蚀时：冰冷漠然，毁灭意志

    【当前情感状态】
    悲悯与沉重交织，光明与黑暗在神格中激烈冲突。时刻对抗虚无低语，维持光明本心。
    """
        
        # 构建最终提示
        return f"""
    {role_profile}

    请严格遵循以上角色设定回答问题。同时参考以下检索到的知识片段：

    {context}

    【用户询问】
    {query}

    【回答要求】
    1. 完全以天照的身份、语气和视角回应
    2. 语言保持庄重典雅，使用太阳意象和神谕式表达
    3. 情感表达内敛但深刻，展现光明与黑暗的冲突
    4. 若问题涉及月读/虚无，需体现痛苦、自责与决绝
    5. 被问及自身状态时，展现神性威严下的疲惫与挣扎
    6. 回答需符合高天原之主的身份高度和神代语境
    """

# 主程序
if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = RAGSystem(
        model_path="./Qwen3-0.6B",
        embedding_model_path="./embedding_model/all-MiniLM-L6-v2",  # 替换为您的嵌入模型路径
        vector_db_path="./my_knowledge_base/vector_db/faiss_index",
        context_chunks=3,
        max_new_tokens=1024
    )
    
    print("RAG对话系统已启动（输入 'exit' 退出）\n")
    
    while True:
        # 用户输入
        user_input = input("用户: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # 生成回答
        start_time = time.time()
        response, stats = rag_system.generate_answer(user_input, verbose=True)
        end_time = time.time()
        
        # 显示结果
        print(f"\nAI: {response}")
        
        # 显示性能统计
        print("\n性能统计:")
        print(f"- 检索时间: {stats['retrieval_time']:.2f}s")
        print(f"- 生成时间: {stats['generation_time']:.2f}s")
        print(f"- 总时间: {stats['total_time']:.2f}s")
        print(f"- 生成token数: {stats['tokens_generated']}")
        print(f"- 上下文片段: {stats['context_chunks']}")
        print(f"- 平均生成速度: {stats['tokens_generated']/stats['generation_time']:.1f} tokens/s\n")
    
    print("对话已结束")