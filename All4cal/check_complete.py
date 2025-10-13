import os
import json
def find_file_listdir(directory, filename):
    """
    使用 os.listdir() 查找文件
    """
    try:
        files = os.listdir(directory)
        return filename in files
    except FileNotFoundError:
        print(f"目录不存在: {directory}")
        return False

p0 = '/work/home/ac877eihwp/renyq/20250828TT/test/jobsub/RDA_Spre/'
fj = '/feedback.json'
dictfeedback = {}
for i in range(10):
    completePath=f'{p0}{i}{fj}'
    with open(completePath,'r') as j:
        jd = json.load(j)
    dictfeedback.update(jd)
cp=0
error = []
warn = []
for reaction in dictfeedback:
    reactiondict = dictfeedback[reaction]
    fmaxIS,fmaxFS=reactiondict['IS_fmax'],reactiondict['FS_fmax']
    IScheck,FScheck=reactiondict['IS'],reactiondict['FS']
    if fmaxFS >0.01 or fmaxIS >0.01:
        error.append(('fmax',reactiondict))
    else:
        if all(IScheck)==True:
            if FScheck[0]==True:
                cp=cp+1
                if FScheck[1]==False:
                    warn.append(('FS2',reaction))
        else:
            error.append(('IS',reactiondict))
print(f'任务:RDA_S IS/FS MLP optimization\n任务完成率:{cp}/{len(dictfeedback)}={cp/len(dictfeedback)*100:.3f}%\nwarn:{warn}\nerror:{error}')

p1 = '/work/home/ac877eihwp/renyq/20250828TT/test/RDA_S/'
fj1 = '/foldername.json'
subp='/IntermediateProcess/results'
FFL=0
error=[]
with open (f'{p1}{fj1}','r') as j:
    FJdict = json.load(j)
for FN in FJdict:
    completePath=f'{p1}{FN}{subp}'
    ffl1 = find_file_listdir(completePath,'TS_RDA_S_RdncCopt.xyz')
    ffl2 = find_file_listdir(completePath,'TS_RDA_S_R1Copt.xyz')
    if ffl1 == True or ffl2 == True:
        FFL=FFL+1
    else:
        FFL=FFL
        error.append(FN)
        
print(f'任务:Output:TS_RDA_S_Rdnc.vasp\n任务完成率:{FFL}/{len(FJdict)}={FFL/len(FJdict)*100:.3f}%\nerror:{error}')
#print(f'任务:Output:TS_RDA_S_RdncCopt.vasp\n任务完成率:{FFL2}/{len(FJdict)}={FFL2/len(FJdict)*100:.3f}%\nerror:{error2}')

