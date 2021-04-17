import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
import albumentations
from albumentations import *
from albumentations.pytorch import ToTensorV2
import cv2
import timm
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    transformed = get_transforms()(image = image_data)
    image = transformed['image']
    with torch.no_grad():
        y_preds1 = model(image[None, ...])
        y_preds2 = model(image[None, ...].flip(-1))
        prediction = (y_preds1.sigmoid().to('cpu').numpy() + y_preds2.sigmoid().to('cpu').numpy()) / 2
        
    return prediction.reshape(-1)

def get_transforms():
        return Compose([
            Resize(512, 512),
            Normalize(
            ),
            ToTensorV2(),
        ])

class ResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

def main():
    model = ResNet200D()
    model.load_state_dict(torch.load("./resnet200d_320_CV9632.pth")['model'])
    model.eval()
    
    st.write("""
             # Incorrect Placement of Chest Catheter Detector
             """
             )
    
    st.write("A Resnet-200D based image classification model to detect incorrect place of catheters on chest X-Rays")
    
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    #
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prediction = import_and_predict(image, model)
        prediction = pd.DataFrame({'ETT-Abnormal' : prediction[0],
                      'ETT-Borderline': prediction[1],
                      'ETT-Normal': prediction[2],
                      'NGT-Abnormal': prediction[3],
                      'NGT-Borderline': prediction[4],
                      'NGT-Incorrectly Imaged': prediction[5],
                      'NGT-Normal': prediction[6],
                      'CVC-Abnormal': prediction[7],
                      'CVC-Borderline': prediction[8],
                      'CVC-Normal': prediction[9],
                      'Swan Ganz Catheter': prediction[10],
                      }
                      , index = [0])
        st.write(prediction)
        for i in range(len(prediction.columns)):
            if(prediction.iloc[0,i] > 0.5):
                if(prediction.columns[i].split('-')[1] in ['Borderline', 'Abnormal']):
                    st.error("POSSIBLE DANGER " + str(prediction.columns[i]))

if __name__ == "__main__":
    main()
