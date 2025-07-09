import os
from PyPDF2 import PdfReader
from docx import Document
from langchain_community.document_loaders import TextLoader
import re

def load_and_save_document(file_path, output_dir="parsed_document"):
    """加载、解析文档，并进行预处理后保存为TXT格式"""
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    
    if os.path.isdir(file_path):
        # 如果是文件夹，递归处理所有文件
        for root, _, files in os.walk(file_path):
            for file in files:
                full_path = os.path.join(root, file)
                try:
                    parsed_text = process_single_file(full_path)
                    preprocessed_text = preprocess_text(parsed_text)
                    save_parsed_data(preprocessed_text, full_path, output_dir)
                except ValueError as e:
                    print(f"跳过不支持的文件: {full_path}。错误: {e}")
    else:
        # 如果是单个文件，直接处理
        parsed_text = process_single_file(file_path)
        preprocessed_text = preprocess_text(parsed_text)
        save_parsed_data(preprocessed_text, file_path, output_dir)

def process_single_file(file_path):
    """解析单个文件并返回文本内容"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    elif ext == '.pdf':
        return parse_pdf(file_path)
    elif ext == '.docx':
        return parse_docx(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

def parse_pdf(file_path):
    """解析PDF文件并返回文本内容"""
    full_text = []
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                full_text.append(text)
    return "\n".join(full_text)

def parse_docx(file_path):
    """解析DOCX文件并返回文本内容"""
    full_text = []
    doc = Document(file_path)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            full_text.append(text)
    return "\n".join(full_text)

def preprocess_text(text):
    """预处理文本，为结构标题添加标签"""
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # 匹配一级标题（如 "一、核心身份信息"）
        match_1 = re.match(r'^([一二三四五六七八九十]+、)(.+)', line)
        # 匹配二级标题（如 "1.角色名称："）
        match_2 = re.match(r'^(\d+\.)(.+)', line)
        
        if match_1:
            # 添加一级标题标签
            processed_line = f"[SECTION_1]{match_1.group(1)}{match_1.group(2)}[/SECTION_1]"
            processed_lines.append(processed_line)
        elif match_2:
            # 添加二级标题标签
            processed_line = f"[SECTION_2]{match_2.group(1)}{match_2.group(2)}[/SECTION_2]"
            processed_lines.append(processed_line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def save_parsed_data(parsed_text, original_path, output_dir):
    """保存解析后的文本到TXT文件"""
    base_name = os.path.basename(original_path)
    file_ext = os.path.splitext(base_name)[1].lower()[1:]  # 去掉点，如 'pdf'
    
    # 创建子目录（如 ./parsed_document/pdf/）
    sub_dir = os.path.join(output_dir, file_ext)
    os.makedirs(sub_dir, exist_ok=True)
    
    output_name = f"{os.path.splitext(base_name)[0]}.txt"
    output_path = os.path.join(sub_dir, output_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(parsed_text)
    
    print(f"解析结果已保存至: {output_path}")

if __name__ == "__main__":
    # 硬编码输入路径和输出路径
    input_path = "./Amaterasu"  # 输入目录
    output_dir = "./parsed_document"  # 输出目录
    
    print(f"开始处理文档目录: {input_path}")
    load_and_save_document(input_path, output_dir)
    print("所有文档处理完成!")