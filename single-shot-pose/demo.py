import argparse
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainercv.utils import read_image
from chainercv.visualizations import vis_image

from lib.ssp import SSPYOLOv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model')
    parser.add_argument('image')
    args = parser.parse_args()

    img = read_image(args.image)
    model = SSPYOLOv2()
    chainer.serializers.load_npz(args.pretrained_model, model)
    bboxes, labels, scores = model.predict([img])
    bbox = bboxes[0]
    label = labels[0]
    score = scores[0]

    for i in range(len(bbox)):
        vis_image(img)
        for j in range(9):
            plt.scatter(bbox[i][j][0], bbox[i][j][1]) 
        plt.show()


if __name__ == '__main__':
    main()
