import numpy as np
import scipy.linalg as alg
import matplotlib.pyplot as plt
import time
import streamlit as st
import pandas as pd
from typing import List, Dict


# reduced from 22 seconds (for rank 100) to 12 seconds by changing from numpy to scipy
# reduced to 6 seconds by doing matrix multiplications all at once

def float_range(start, stop, step) -> List:
    """
    Create range of floating numbers

    :param start: first floating number
    :param stop: maximum floating number
    :param step: amount by which to increment floating numbers
    :return:
    """

    result = []
    current = start
    while current < stop:
        result.append(current)
        current += step

    return result


class SvdResult():
    """
    Class to calculate and store svd results
    """

    def __init__(self, array):
        """

        :param array: initial arrays to which to apply SVD
        """
        self.U, self.sigma, self.V_t = alg.svd(array)


class Image():
    """

    """

    def __init__(self, img):
        """
        Class to store image arrays and initialize approximation dictionaries
        :param img: path to image, or numpy array containing image data
        """
        # if img is a string, load data, else store image data
        if isinstance(img, str):
            self.img = plt.imread(img)
        else:
            self.img = img

        # store individual arrays of each dimension of image
        self.arrays = {i: self.img[:, :, i] for i in range(self.img.shape[2])}
        self.array_approxs = {}

    @st.cache(persist=True)
    def get_image_svd(self) -> Dict[int, np.ndarray]:
        """
        Calculate and return svd arrays for each dimension of image
        :return:
        """
        return {i: SvdResult(self.arrays[i]) for i in range(self.img.shape[2])}

    # @st.cache
    def calculate_desired_rank(self, perc):
        """
        Calculate desired rank of approximation, given desired percentage of storage

        :param perc: percentage of storage to be maintained in approximation
        :return:
        """
        self.desired_rank = round((self.img.shape[0] * self.img.shape[1]) / (self.img.shape[0] + self.img.shape[1]) * perc / 100)

    def add_ranks(self, svd_results) -> np.ndarray:
        """
        build approximation by adding up matrices up to desired rank

        :param svd_results: svd results of image
        :return:
        """

        for i in range(self.img.shape[2]):
            self.array_approxs[i] = svd_results[i].sigma[:self.desired_rank] * svd_results[i].U[:, :self.desired_rank] @ svd_results[i].V_t[
                                                                                                   :self.desired_rank]

        return np.concatenate([i[:, :, np.newaxis] for i in self.array_approxs.values()], axis=2) / 255

    # @st.cache
    def get_stats(self, svd_results, maximum=25)->pd.DataFrame:
        """
        Calculate and return dataframe of relative storage spaces and information capture rates

        :param svd_results: svd_results of image
        :param maximum: maximum percentage of original storage size for which to calculate information capture
        :return:
        """

        # percentage of storage represented by each rank compared to original storage
        step = 100 * (self.img.shape[0] + self.img.shape[1]) / (self.img.shape[0] * self.img.shape[1])

        # get list of storage sizes for increasing rank numbers
        self.storage = float_range(0, maximum, step)

        # cumulative percentage of total singular values represented using each successive rank
        self.captures = {i: 100 * np.cumsum(svd_results[i].sigma[:len(self.storage)]) / np.sum(svd_results[i].sigma[:])
                         for i in svd_results}

        df = pd.DataFrame({'Storage (%)': self.storage})
        for i in self.captures:
            df[f'Channel {i+1}'] = self.captures[i]
        return df


def read_and_show_img(img_path):
    """
    Store image and plot

    :param img_path: path to image to be plotted
    :return:
    """
    img = plt.imread(img_path)
    plt.imshow(img)


def do_everything(img_path, perc=60):
    """

    :param img_path:
    :param perc:
    :return:
    """
    img = Image(img_path)
    svd_results = img.get_image_svd()
    img.calculate_desired_rank(perc=perc)
    img.add_ranks(svd_results)
    img.get_stats(svd_results)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    do_everything(perc=100)
    print(time.time() - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
