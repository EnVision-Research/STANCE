import os

def find_mp4_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    f.write(full_path + '\n')

directory = 'YOUR/OWN/VIDEO/PATH'  # 你要遍历的文件夹路径
output_file = 'YOUR_VIDEO.txt'  # 输出的 .txt 文件路径

find_mp4_files(directory, output_file)
print(f"所有 .mp4 文件的路径已导出到 {output_file}")