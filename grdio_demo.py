import torch
import gradio as gr


# 加载模型
# torch.hub.load函数参数及用法
model = torch.hub.load("./","custom",path = "/home/udf/workspace/wangzijian/yolov5/yolov5s.pt",source= "local")


# 增加描述
title = "基于Yolov5的演示项目"

desc = "基于Yolov5的演示项目非常简洁和方便"

def det_image(img,conf_thres,iou_thres):
    model.conf = conf_thres 
    model.iou = iou_thres 
    return model(img).render()[0]

# 可以自己指定开始的阈值标准
base_conf = 0.25
base_iou = 0.45
show_conf ,show_iou= 0.2, 0.4

# 可以将"image"参数改为gr.webcam()就可以实现摄像头的调用和检测
# live参数可以使得不按提交按钮可以自动生成检测
# 利用launch里面的share参数可以实现本地转公网地址

gr.Interface(inputs=["image",gr.Slider(minimum = 0,maximum = 1,value = base_conf),gr.Slider(minimum = 0,maximum = 1,value = base_iou)],
             outputs=["image"],
             fn = det_image,
             title = title,
             description = desc,
             live = True,
             examples=[["/home/udf/workspace/wangzijian/yolov5/data/images/bus.jpg",show_conf,show_iou],["/home/udf/workspace/wangzijian/yolov5/data/images/zidane.jpg",base_conf,base_iou]]).launch(share=True)
