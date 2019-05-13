import numpy as np
from skimage import io, color, img_as_ubyte, segmentation, filters
from skimage.color import rgb2gray
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.spatial import distance

from scipy import ndimage as ndi

import cv2
import math

import warnings
warnings.filterwarnings('ignore')

def show2imgs(im1, im2, title1='Obraz pierwszy', title2='Obraz drugi', size=(10, 10)):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    ax1.imshow(im1, cmap='gray')
    ax1.axis('off')
    ax1.set_title(title1)

    ax2.imshow(im2, cmap='gray')
    ax2.axis('off')
    ax2.set_title(title2)
    plt.show()


def getFigure(labelledImage, objNumber):
    points = []
    for y in range(labelledImage.shape[0]):
        for x in range(labelledImage.shape[1]):
            if labelledImage[y, x] == objNumber:
                points.append((y, x))

    return points


def cog2(points):
    mx = 0
    my = 0
    for (y, x) in points:
        mx = mx + x
        my = my + y
    mx = mx / len(points)
    my = my / len(points)

    return [my, mx]


def computeBB(points):
    s = len(points)
    my, mx = cog2(pts)

    r = 0
    for point in points:
        r = r + distance.euclidean(point, (my, mx)) ** 2

    return s / (math.sqrt(2 * math.pi * r))


def computeFeret(points):
    px = [x for (y, x) in points]
    py = [y for (y, x) in points]

    fx = max(px) - min(px)
    fy = max(py) - min(py)

    return float(fy) / float(fx)


def getImage(url):
    image = io.imread(url)
    image = color.rgb2gray(image)
    image = img_as_ubyte(image)

    return image


# obróbka obrazka
def obrobka(im):
    im = img_as_ubyte(rgb2gray(im))

    im = cv2.medianBlur(im, 11)
    val = filters.threshold_otsu(im)

    image_przed = ndimage.binary_fill_holes(im < val)

    label_objects, nb_labels = ndi.label(image_przed)
    sizes = np.bincount(label_objects.ravel())

    # sprawdzenie, czy których obiekt nie jest sklejony z innym
    # (obrazki robione z tej samej wysokości, więc nie mogą być większe niż pewna wartosć)
    def sprawdzenie(image):
        image_sprawdzenie = image
        for i in range(1, len(sizes)):
            if sizes[i] > 70240:
                kernel = np.ones((11, 11), np.uint8)
                image_sprawdzenie = image_sprawdzenie * 255.0
                image_sprawdzenie = cv2.erode(image_sprawdzenie, kernel, iterations=13)
                image_sprawdzenie = cv2.dilate(image_sprawdzenie, kernel, iterations=6)
                return image_sprawdzenie
        return image

    image_po = sprawdzenie(image_przed)
    label_objects, nb_labels = ndi.label(image_po)
    sizes = np.bincount(label_objects.ravel())

    show2imgs(image_przed, image_po)
    po = (np.sum(image_po > 0))
    przed = np.sum(image_przed > 0)
    print('Ile pikseli obiektów przed operacjami: ' + str(przed))
    print('Ile piskeli obiektów po operacjach: ' + str(po))
    print('Dokładność: ' + str(po * 100 / przed))
    print("Ile obiektów na obrazku: " + str(len(sizes) - 1))
    img_size = image_przed.shape[0] * image_przed.shape[1]
    size_all = 0
    for i in range(1, len(sizes)):
        print('Obrazek nr ' + str(i) + " ma " + str(sizes[i]) + ' pikseli i zajmuje {:.2f} % obiektu.'.format(
            sizes[i] * 100 / img_size))
        size_all += sizes[i] * 100 / img_size
    print(size_all)
    return nb_labels, label_objects

#część główna programu
image = io.imread('7.jpg')
nb_labels, label_objects = obrobka(image)

for i in range(nb_labels):
    print("\nObiekt numer" + " " + str(i + 1))
    pts = getFigure(label_objects, i + 1)
    bb = computeBB(pts)
    feret = computeFeret(pts)

    print('Liczba punktow: ', len(pts), '\nSrodek ciezkosci: ', cog2(pts), '\nBlair-Bliss: ', bb, '\nFeret: ', feret,
          '\n---\n')