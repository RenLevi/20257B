import SUPPORT.support as sp
import json
import os
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    reaction_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            reaction_list.append(string)
        reaction_list.pop(0)
    return reaction_list
def batch_process(data, batch_size):
    for i in range(0, len(data), batch_size):


        yield data[i:i + batch_size]
i = 1
fdl = read_file_line_by_line('/public/home/ac877eihwp/renyq/C2/test/RN/reactionslist.txt')
FLODER = 'RDA_S'
for batch in batch_process(fdl,11):
    sp.run_command_in_directory(directory=f'test/jobsub/{FLODER}/{i}',command='qsub jobRDA_S.sh')
    i += 1