import numpy as np
import random
from main_new import *
import streamlit as st
import PIL
import random


@st.cache(suppress_st_warning=True)
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
    #link stylesheet and include introduction text
    st.markdown('<link rel="stylesheet" href=styles.css>', unsafe_allow_html=True)
    st.header('IMAGE COMPRESSION')
    st.markdown('<p style="color:blue">Welcome to this demonstration of image compression,'
                ' using Singular Value Decomposition.</p>', unsafe_allow_html=True)

    #include image picker
    image = st.file_uploader(label='Input an image to begin compression.', type=['png', 'jpg'])

    #once image is selected, display image and chart depicting possible compression limits
    if image is not None:
        raw_image = PIL.Image.open(image)
        np_img = np.asarray(raw_image)
        proc_image = Image(img=np_img)
        svd_results = proc_image.get_image_svd()
        st.image(image)
        df = proc_image.get_stats(maximum=50, svd_results=svd_results)
        st.subheader('Information Captured (%) vs Storage Space Used (%)')
        st.line_chart(data=df, x='Storage (%)')

        # include percentage picker.
        perc = st.slider("Select a percentage for storage usage. "
                         "(The smaller the storage, the lower quality the compressed picture will have.)",
                         1.0, 100.0, value=10.0, step=0.5)

        #process image and display choice
        proc_image.calculate_desired_rank(perc=perc)
        plotting = proc_image.add_ranks(svd_results=svd_results)
        sample = random_sample(raw_image, processed_image=plotting)
        st.markdown('One of the below pictures has been compressed. Which is it?')
        for index, column in enumerate(st.columns(4)):
            with column:
                st.image(image=sample[index], clamp=True)
                if st.button('SELECT', key=index):

                    if type(sample[index]) == np.ndarray:
                        st.markdown('<p style="color:green">CORRECT</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="color:red">INCORRECT</p>', unsafe_allow_html=True)
