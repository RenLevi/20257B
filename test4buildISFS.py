import build_ISFS.pre4SearchTS as pre4TS
#NEB搜索过渡态#并列执行
Pre4TS = pre4TS.PREforSearchTS('model/')
Pre4TS.site_finder()
Pre4TS.readDataPath()
Pre4TS.buildmodel('model/RN/reactionslist.txt','prototypeModel.pth')