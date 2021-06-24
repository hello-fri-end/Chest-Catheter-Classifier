# Multi Label Classificaion using Resnet200D
A dockerised web app made using streamlit that predicts incorrect placement of chest-catheters using X-Ray images.

## Description

Refer https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview


## How to deploy the Streamlit app locally with Docker 
Assuming docker is running on your machine and you have a docker account, do the following:
- Build the image

``` bash
docker build -t <USERNAME>/<YOUR_IMAGE_NAME> .
```

- Run the image

``` bash
docker run -p 8501:8501 <USERNAME>/<YOUR_IMAGE_NAME>
```

- Open your browser and go to `http://localhost:8501/`


## How to deploy the Streamlit app locally without Docker
- Install the dependencies 
```bash
pip install -r requirements.txt
```
- Run the Streamlit app
```bash
streamlit run app.py
```


### Demonstration

<a href="https://imgflip.com/gif/59y81c"><img src = "https://i.imgflip.com/59y81c.gif" title = "Classifier"/></a>
