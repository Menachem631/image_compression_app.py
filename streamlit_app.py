import numpy as np
import random
from main_new import *
import streamlit as st
import PIL
import random


@st.cache
def random_sample(raw_image, processed_image):
    """
    Shuffle the order of approximate image and 3 copies of original image

    :param raw_image: original image
    :param processed_image: approximate image
    :return:
    """
    result = [raw_image, raw_image, raw_image, processed_image]
    return random.sample(result, 4, )


if __name__ == '__main__':
    st.markdown('<link rel="stylesheet" href=styles.css>', unsafe_allow_html=True)
    st.header('IMAGE COMPRESSION')
    st.markdown('<p style="color:blue">Welcome to this demonstration of image compression,'
                ' using Singular Value Decomposition</p>', unsafe_allow_html=True)

    image = st.file_uploader(label='Input an image to begin compression', type=['png', 'jpg'])

    if image is not None:
        raw_image = PIL.Image.open(image)
        np_img = np.asarray(raw_image)
        proc_image = Image(img=np_img)
        svd_results = proc_image.get_image_svd()
        st.image(image)
        df = proc_image.get_stats(maximum=50, svd_results=svd_results)
        st.subheader('Information Captured (%) vs Storage Space Used (%)')
        st.line_chart(data=df, x='Storage (%)')

        perc = st.slider("Select a percentage for storage usage. "
                         "(The smaller the storage, the lower quality the compressed picture will have.)",
                         1.0, 50.0, value=10.0, step=0.5)

        proc_image.calculate_desired_rank(perc=perc)
        plotting = proc_image.add_ranks(svd_results=svd_results)
        sample = random_sample(raw_image, processed_image=plotting)
        st.markdown('One of the below pictures has been compressed. Which is it?')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(image=sample[0], clamp=True)
            if st.button('SELECT', key=0):

                if type(sample[0]) == np.ndarray:
                    st.markdown('<p style="color:green">CORRECT</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:red">INCORRECT</p>', unsafe_allow_html=True)
        with col2:
            st.image(image=sample[1], clamp=True)
            if st.button('SELECT', key=1):
                if type(sample[1]) == np.ndarray:
                    st.markdown('<p style="color:green">CORRECT</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:red">INCORRECT</p>', unsafe_allow_html=True)
        with col3:
            st.image(image=sample[2], clamp=True)
            if st.button('SELECT', key=2):
                if type(sample[2]) == np.ndarray:
                    st.markdown('<p style="color:green">CORRECT</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:red">INCORRECT</p>', unsafe_allow_html=True)
        with col4:
            st.image(image=sample[3], clamp=True)
            if st.button('SELECT', key=3):
                if type(sample[3]) == np.ndarray:
                    st.markdown('<p style="color:green">CORRECT</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:red">INCORRECT</p>', unsafe_allow_html=True)
