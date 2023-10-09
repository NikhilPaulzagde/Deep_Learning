from pyngrok import ngrok
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image

model= pickle.load(open('svm_model.pkl','rb'))


# Define a function to add custom CSS style
def set_css_style():
    st.markdown(
        """
        <style>
        /* Customize the color of the progress bar */
        .stProgress > div > div > div {
            background-color: #525050 !important; /* Replace  #292727 with your desired color */
        }
        /* Center align the elements in the app */
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .stButton > button {
            width: 100%;
            max-width: 150px;
            margin: 10px auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the function to set custom CSS style
set_css_style()



st.title("Flower Classifier App")
st.write("Upload an image and click 'PREDICT' to classify the flower!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("PREDICT"):
        Categories = ["Jasmine", "Rose", "Sunflower"]
        st.write('Result...')
        flat_data = []
        img = np.array(img)
        img_resized = resize(img, (150, 150, 3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = Categories[y_out[0]]
        st.title(f'PREDICTED OUTPUT: {y_out}')
        q = model.predict_proba(flat_data)

        for index, item in enumerate(Categories):
            st.write(f'{item} : {q[0][index] * 100:.2f}%')
            st.progress(q[0][index])






