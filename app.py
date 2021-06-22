import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from albumentations import *
from albumentations.pytorch import ToTensorV2
import cv2
import timm
from PIL import Image, ImageOps
from utils import * 
from model import ResNet200D


def main():
    model = ResNet200D()
    model.load_state_dict(torch.load("./resnet200d.pth")['model'])
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
