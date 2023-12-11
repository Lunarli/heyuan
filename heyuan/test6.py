import os
import re
from glob import glob

def clean_srt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                cleaned_line = remove_braces_content(line)
                file.write(cleaned_line)

        print(f"Processed file: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def remove_braces_content(line):
    # 使用正则表达式去除 {} 中的内容
    return re.sub(r'\{.*?\}', '', line)

def process_folder(folder_path):
    try:
        # 获取当前文件夹下所有的.srt文件
        srt_files = glob(os.path.join(folder_path, '*.srt'))

        for srt_file in srt_files:
            clean_srt_file(srt_file)

        # 递归处理子文件夹
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for subfolder in subfolders:
            process_folder(subfolder)

    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")

if __name__ == "__main__":
    root_folder_path = r'G:\video\电影\1-6季电视剧（可调3种字幕：中英文、纯英文、无字幕）'  # 替换为实际的根文件夹路径
    process_folder(root_folder_path)
