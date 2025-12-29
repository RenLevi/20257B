import build_ISFS.pre4SearchTS as pre4TS
import json
with open('config.json','r') as f:
    config = json.load(f)

#NEB搜索过渡态#并列执行
print('-'*10,"start",'-'*20)
Pre4TS = pre4TS.PREforSearchTS('/public/home/ac877eihwp/renyq/C2/test/',I=config['INAME'])
print('-'*10,"start site finder",'-'*8)
Pre4TS.site_finder()
print('-'*10,"start readData",'-'*11)
Pre4TS.readDataPath()
print('-'*10,"start build model",'-'*8)
Pre4TS.buildmodel(config['folderpath'],MLPs_model_path=config['MLPs_model_path'])