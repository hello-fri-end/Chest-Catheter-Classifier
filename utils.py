def import_and_predict(image_data, model):
    """
    This function takes an image & performs a forward
    pass through the model, returning the model's predictions
    on the image.
    """
    transformed = get_transforms()(image = image_data)
    image = transformed['image']
    with torch.no_grad():
        y_preds1 = model(image[None, ...])
        y_preds2 = model(image[None, ...].flip(-1))
        prediction = (y_preds1.sigmoid().to('cpu').numpy() + y_preds2.sigmoid().to('cpu').numpy()) / 2
        
    return prediction.reshape(-1)

def get_transforms():
    """
    This functions returns an object that applies
    various transformations to the images.
    """
    return Compose([
            Resize(512, 512),
            Normalize(
            ),
            ToTensorV2(),
        ])

