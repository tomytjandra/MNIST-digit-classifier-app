# image processing
from PIL import Image
import numpy as np

# modelling
import torch
import torch.nn as nn
import torchvision.transforms as T

# LOAD PRETRAINED MODEL
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
    )
model.load_state_dict(torch.load('model.pt'))

# PREDICT FUNCTION
def predict_proba(image):
    # process image
    image = Image.fromarray(image.astype(np.uint8)).convert('L')
    image = image.resize((28, 28))
    
    # convert to tensor
    transforms = T.ToTensor()
    tensor = transforms(image)

    # if canvas is not drawn
    if tensor.min().item() == tensor.max().item():
        return image, None

    # predict
    with torch.no_grad():
        output = model(tensor.view(1,-1))
        prob = torch.exp(output)
    return image, prob.view(-1)