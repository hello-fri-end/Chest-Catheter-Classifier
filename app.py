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
    # Load the trained model
    model = ResNet200D()
    model.load_state_dict(torch.load("./resnet200d.pth")['model'])
    model.eval()
    
    # Heading
    st.write("""
             # Incorrect Placement of Chest Catheter Detector
             """
             )
    
    # Description
    st.write("A Resnet-200D based image classification model to detect incorrect place of catheters using chest X-Rays")
    
    # Prompt to upload the X-Ray Image
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        # Apply some basic transformations
        image = Image.open(file)
        st.image(image, use_column_width=True)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Do a forward pass through the model
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
        # Print the probablities
        st.write(prediction)
        # If any of the catheter's has > 50% probablity in 
        # borderline or abnormal category, alert the user.
        for i in range(len(prediction.columns)):
            if(prediction.iloc[0,i] > 0.5):
                if(prediction.columns[i].split('-')[1] in ['Borderline', 'Abnormal']):
                    st.error("POSSIBLE DANGER " + str(prediction.columns[i]))

if __name__ == "__main__":
    main()
