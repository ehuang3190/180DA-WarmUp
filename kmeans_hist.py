#source: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


#source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
#https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

'''
Improvements:
add plt.pause(0.1) to allow for realtime calculation of k-means
'''

import numpy as np
import cv2 as cv


cap = cv.VideoCapture(0)

while(True):

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    # Take each frame
    _, frame = cap.read()
    cv.imshow('frame',frame)
    img = frame[80:160, 150:230]
    cv.imshow('image',img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3, n_init=10) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    #plt.show()
    plt.pause(0.1)
    plt.clf()


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()