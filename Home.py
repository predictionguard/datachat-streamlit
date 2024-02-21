import streamlit as st
from sentence_transformers import SentenceTransformer
import base64
import pandas as pd


#---------------------#
# Streamlit Styling   #
#---------------------#
# Apply custom styles using CSS
st.markdown(
    """
    <style>
        /* Main content background to black and text color to white for contrast */
        body {
            color: #FFFFFF; /* White text color */
            background-color: #000000; /* Black background color */
        }
        .stApp {
            color: #FFFFFF; /* White text color */
            background-color: #000000; /* Black background color */
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #FFFFFF; /* White background for the sidebar */
            color: #000000; /* Black text color for contrast */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#---------------------#
# Load embeddings     #
#---------------------#

# Load model
@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

st.session_state['en_emb'] = load_model("all-MiniLM-L12-v2")
# st.session_state['multi_emb'] = load_model("stsb-xlm-r-multilingual")

#---------------------#
#    Main Page        #
#---------------------#

st.title("Prediction Guard Chat Assistant")
st.markdown("Explore our chat with data demo using privacy-conserving AI models from [Prediction Guard](https://www.predictionguard.com/)")
