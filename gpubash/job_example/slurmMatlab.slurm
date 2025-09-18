#!/bin/bash
#SBATCH -J vasp                       #作业脚本名称             自定义
#SBATCH -p kshctest                   #队列名称                 AC平台首页有"可访问队列"一栏
#SBATCH -N 1                          #使用计算节点数量         根据实际需要修改
#SBATCH --ntasks-per-node=30          #每个计算节点使用30核心   不建议修改，占满32核心运行效率比较低
#SBATCH --mem=50g                     #申请内存                 根据实际需要修改
##SBATCH --exclusive                  #作业独占参数             占满30核心建议去掉注释符号使独占生效，提高效率
matlab -nodesktop -nosplash -nodisplay -r "test_case"
