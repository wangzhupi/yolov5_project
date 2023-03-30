import torch
import gradio as gr


# 模型加载的函数
def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model.eval()
    return model

