import os, sys
import shutil

# 设置源目录和目标目录
source_dir = '/data/sxj/Segment-Anything-in-4D/data/hypernerf/interp/aleks-teapot/rgb/2x'
target_dir = '/data/sxj/Segment-Anything-in-4D/data/hypernerf/interp/aleks-teapot/rgb/demo'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取源目录下的所有文件
all_files = os.listdir(source_dir)

# 对文件进行排序，这个排序的方式取决于你的文件命名规则
# 如果你的文件是按照数字顺序命名的，那么你可以使用以下代码进行排序：
all_files.sort(key=lambda x: int(x[:-4]))  # Assumes file format is '.jpg'

start_index = all_files.index('000100.png')
end_index = all_files.index('000320.png') + 1

# 提取第1900-2200张图片
selected_files = all_files[start_index:end_index]  # Python indexing starts at 0

# sys.exit(0)
# 复制选定的文件到目标目录
for file_name in selected_files:
    source = os.path.join(source_dir, file_name)
    target = os.path.join(target_dir, file_name)
    shutil.copyfile(source, target)

print("Images have been moved successfully.")