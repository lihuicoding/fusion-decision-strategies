# import os
# import pandas as pd
#
# path = r'dat_file'  # 旧文件存放目录目录
# path_new = r'dataset'  # 新文件存放的目录
#
# filelist = os.listdir(path)  # 目录下所有的文件列表
#
# for files in filelist:
#     yuan_path = os.path.join(path, files)
#     file_name = os.path.splitext(files)[0]  # 文件名
#     Newdir = os.path.join(path_new, str(file_name) + '.csv')
#     data = []
#     with open(yuan_path, 'r', encoding='utf-8-sig') as df:
#         for line in df:
#             data.append(list(line.strip().split()))
#     dataset = pd.DataFrame(data)
#     dataset.to_csv(Newdir, index=None)

import os

path_0 = r"dat_file"  # 原文件目录
path_1 = r"dataset"  # 存放目录
filelist = os.listdir(path_0)  # 目录下文件列表
for files in filelist:
    dir_path = os.path.join(path_0, files)
    # 分离文件名和文件类型
    file_name = os.path.splitext(files)[0]  # 文件名
    file_type = os.path.splitext(files)[1]  # 文件类型
    # 将.dat文件转为.csv文件
    if file_type == '.dat':  # 可切换为.xls等
        file_test = open(dir_path, 'rb')  # 读取原文件
        new_dir = os.path.join(path_1, str(file_name) + '.csv')
        # print(new_dir)
        file_test2 = open(new_dir, 'wb')  # 创建/修改新文件
        for lines in file_test.readlines():
            lines = lines.decode()
            str_data = ",".join(lines.split(' '))  # 分隔符依据自己的文件确定
            file_test2.write(str_data.encode("utf-8"))
        file_test.close()
        file_test2.close()
