# image processing
from PIL import Image
import numpy as np

# visualization
import matplotlib.image as mimage
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# modelling
import torch
import torch.nn as nn
import torchvision.transforms as T

# web app
import streamlit as st
from streamlit_drawable_canvas import st_canvas

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

# PAGE CONFIGURATION
st.set_page_config(
    page_title = "Handwritten Digits Classifier",
    page_icon = ":pencil:",
)
st.markdown("""
            <style>            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}            
            </style>            
            """, unsafe_allow_html=True)

# TITLE
st.markdown("<h1 style='text-align: center;'>Handwritten Digits Classifier</h1>", unsafe_allow_html=True)

# SIDEBAR
stroke_width = st.sidebar.slider("Stroke width: ", 25, 35, 30)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")

# CONTENT
st.markdown('#')
col1, col2 = st.beta_columns(2)

with col1:
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if canvas_result.image_data is not None:
        # get predicted probability
        image, prob = predict_proba(canvas_result.image_data)
        
        # visualize
        fig, ax = plt.subplots(figsize=(5, 4.5))
        if prob is not None:
            ax.barh(range(0,10), prob, color="blue")
            ax.barh(torch.argmax(prob), torch.max(prob), color="green")
            ax.text(0.5 * torch.max(prob), torch.argmax(prob), "PREDICTION",
                    ha="center", va="center", color="white")
        ax.invert_yaxis()
        ax.set_yticks(range(0,10))
        ax.set_xlabel("Predicted Probability", fontweight="bold")
        plt.tight_layout()

        # st.image(canvas_result.image_data)
        # st.image(image)
        st.pyplot(fig)
        # st.plotly_chart(fig)