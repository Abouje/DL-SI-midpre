import base64
b=lambda s:base64.b64decode(s).decode("utf-8")
path="/workspace/HW2/HW2_"+b("5L2c5Lia5oql5ZGK")+".md"
src=open(path,encoding="utf-8").read()
old1=b("5Zyo6aqM6K+B6ZuG5Zu+54mH5LiK55qE5Y2V5bin5qOA5rWL6KGo546w5q2j5bi477yM5qih5Z6L6IO96K+G5Yir5Ye66YGT6Lev5Zy65pmv5Lit55qE5aSa57G76L2m6L6G55uu5qCH77yI6L2/6L2m44CB5Y2h6L2m44CB5pGp5omY6L2m562J77yJ77yM5YWxIDEyMTYg5p2h5qOA5rWL6K6w5b2V77yM5bmz5Z2H5q+P5bin57qmIDkg5Liq55uu5qCH5qGG77yM5qih5Z6L6IO956iz5a6a6K+G5Yir6L2/6L2m44CB5Y2h6L2m44CB5pGp5omY6L2m562J5aSa57G76L2m6L6G44CC")
new1=b("MTM2IOW4p+WFqOmDqOS6p+eUn+acieaViOajgOa1i++8jOWFsSAxMjE2IOadoeajgOa1i+iusOW9le+8jOW5s+Wdh+avj+W4p+e6piA5IOS4quebruagh+ahhu+8jOaooeWei+iDveeos+WumuivhuWIq+i9v+i9puOAgeWNoei9puOAgeaRqeaJmOi9puetieWkmuexu+i9pui+huOAgg==")
src=src.replace(old1,new1)
lines=src.split(chr(10))
det="检测效果"
trk="跟踪稳定性"
det_new="- **"+det+"**：136 帧全部产生有效检测，共 1216 条检测记录，平均每帧约 9 个目标框，模型能稳定识别轿车、卡车、摩托车等多类车辆。"
trk_new="- **"+trk+"**：共产生 15 个唯一 Track ID；ID=3 持续跟踪 111 帧（占视频 82%），ID=6 持续 96 帧，轨迹 ID 稳定连续，ByteTrack 卡尔曼滤波在真实视频上运动预测正常工作。"
for i,l in enumerate(lines):
  if "**"+det+"**" in l: lines[i]=det_new
  elif "**"+trk+"**" in l: lines[i]=trk_new
src=chr(10).join(lines)
open(path,"w",encoding="utf-8").write(src)
print("done",len(src))
