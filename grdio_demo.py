import torch
import gradio as gr

# 模型加载
# 这个模型加载的函数是利用torch的函数来进行的
# 后面搜一下这些参数的含义以及用法
model = torch.hub.load("./","custom",path = "path/to/weights",source="local)
 
# 增加网页描述信息
title = "基于Yolov5的演示项目"
desc = "基于Yolov5的演示项目非常简洁和方便"

# 检测函数传入基本的conf iou 以及img
def det_image(img,conf_thres,iou_thres):
    model.conf = conf_thres 
    model.iou = iou_thres 
    return model(img).render()[0]

# 可以自己指定开始的阈值标准
base_conf = 0.25
base_iou = 0.45
show_conf ,show_iou= 0.2, 0.4

# 可以将"image"参数改为gr.webcam()就可以实现摄像头的调用和检测
# 这里examples参数用来提供两个demo
gr.Interface(inputs=["image",gr.Slider(minimum = 0,maximum = 1,value = base_conf),gr.Slider(minimum = 0,maximum = 1,value = base_iou)],
             outputs=["image"],
             fn = det_image,
             title = title,
             description = desc,
             examples=[["/home/udf/workspace/wangzijian/yolov5/data/images/bus.jpg",show_conf,show_iou],["/home/udf/workspace/wangzijian/yolov5/data/images/zidane.jpg",base_conf,base_iou]]).launch()
