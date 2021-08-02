# visualization
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# modelling
import torch
from model import predict_proba

# web app
import streamlit as st
from streamlit_drawable_canvas import st_canvas

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