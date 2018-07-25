from __future__ import division
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links import Conv2DBNActiv

from chainercv.transforms import resize


def reorg(x, stride=2):
    stride = 2
    B, C, H, W = x.shape
    assert(H % stride == 0)
    assert(W % stride == 0)
    x = x.reshape(B, C, H // stride, stride, W // stride, stride)
    x = x.transpose((0, 1, 2, 4, 3, 5))
    x = x.reshape(B, C, H // stride * W // stride, stride * stride)
    x = x.transpose((0, 1, 3, 2))
    x = x.reshape(B, C, stride * stride, H // stride, W // stride)
    x = x.transpose((0, 2, 1, 3, 4))
    x = x.reshape(B, stride * stride * C, H // stride, W // stride)
    return x


def leaky_relu(x):
    return F.leaky_relu(x, slope=0.1)


class SSPYOLOv2(chainer.Chain):

    def __init__(self, n_class=1):
        self.n_class = n_class

        super(SSPYOLOv2, self).__init__()
        kwargs = {'activ': leaky_relu, 'bn_kwargs': {'eps': 1e-4}}
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(3, 32, 3, 1, 1, **kwargs)
            self.conv2 = Conv2DBNActiv(32, 64, 3, 1, 1, **kwargs)
            self.conv3 = Conv2DBNActiv(64, 128, 3, 1, 1, **kwargs)
            self.conv4 = Conv2DBNActiv(128, 64, 1, 1, 0, **kwargs)
            self.conv5 = Conv2DBNActiv(64, 128, 3, 1, 1, **kwargs)
            self.conv6 = Conv2DBNActiv(128, 256, 3, 1, 1, **kwargs)
            self.conv7 = Conv2DBNActiv(256, 128, 1, 1, 0, **kwargs)
            self.conv8 = Conv2DBNActiv(128, 256, 3, 1, 1, **kwargs)
            self.conv9 = Conv2DBNActiv(256, 512, 3, 1, 1, **kwargs)
            self.conv10 = Conv2DBNActiv(512, 256, 1, 1, 0, **kwargs)
            self.conv11 = Conv2DBNActiv(256, 512, 3, 1, 1, **kwargs)
            self.conv12 = Conv2DBNActiv(512, 256, 1, 1, 0, **kwargs)
            self.conv13 = Conv2DBNActiv(256, 512, 3, 1, 1, **kwargs)
            self.conv14 = Conv2DBNActiv(512, 1024, 3, 1, 1, **kwargs)
            self.conv15 = Conv2DBNActiv(1024, 512, 1, 1, 0, **kwargs)
            self.conv16 = Conv2DBNActiv(512, 1024, 3, 1, 1, **kwargs)
            self.conv17 = Conv2DBNActiv(1024, 512, 1, 1, 0, **kwargs)
            self.conv18 = Conv2DBNActiv(512, 1024, 3, 1, 1, **kwargs)
            self.conv19 = Conv2DBNActiv(1024, 1024, 3, 1, 1, **kwargs)
            self.conv20 = Conv2DBNActiv(1024, 1024, 3, 1, 1, **kwargs)

            self.conv21 = Conv2DBNActiv(512, 64, 1, 1, 0, **kwargs)
            self.conv22 = Conv2DBNActiv(1280, 1024, 3, 1, 1, **kwargs)
            self.conv23 = L.Convolution2D(1024, 20, 1, 1, 0, nobias=False)

    def __call__(self, x):
        h = self.conv1(x)
        print('1', h[0, 0, 0, 0])

        h = F.max_pooling_2d(h, 2, 2)
        print('2', h[0, 0, 0, 0])
        h = self.conv2(h)
        print('3', h[0, 0, 0, 0])

        h = F.max_pooling_2d(h, 2, 2)
        print('4', h[0, 0, 0, 0])
        h = self.conv5(self.conv4(self.conv3(h)))
        print('7', h[0, 0, 0, 0])

        h = F.max_pooling_2d(h, 2, 2)
        print('8', h[0, 0, 0, 0])
        h = self.conv8(self.conv7(self.conv6(h)))
        print('11', h[0, 0, 0, 0])

        h = F.max_pooling_2d(h, 2, 2)
        print('12', h[0, 0, 0, 0])
        h = self.conv13(self.conv12(self.conv11(
            self.conv10(self.conv9(h)))))
        print('17', h[0, 0, 0, 0])

        h1 = reorg(self.conv21(h))

        h = F.max_pooling_2d(h, 2, 2)
        print('18', h[0, 0, 0, 0])
        h = self.conv20(self.conv19(self.conv18(
            self.conv17(self.conv16(self.conv15(self.conv14(h)))))))
        print('25', h[0, 0, 0, 0])

        h = F.concat((h1, h))
        print('third from last', h[0, 0:10, 0, 0])
        h = self.conv22(h)
        print('second from last', h[0, 0, 0, 0])
        h = self.conv23(h)
        print('last', h[0, 0, 0, 0])
        return h

    def predict(self, imgs):
        img_W = 544
        img_H = 544
        original_Ws = []
        original_Hs = []
        xs = []
        for img in imgs:
            original_Ws.append(img.shape[2])
            original_Hs.append(img.shape[1])
            img = resize(img, (img_H, img_W))
            x = img[None] / 255  # scale weights in the future
            xs.append(x)
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            network_outputs = self.__call__(self.xp.array(x)).data
            bboxes, labels, scores = self._get_region_boxes(
                network_outputs, img_W, img_H)

        for bbox, original_W, original_H in zip(
                bboxes, original_Ws, original_Hs):
            scale_x = original_W / img_W
            scale_y = original_H / img_H

            bbox[:, :, 0] *= scale_x
            bbox[:, :, 1] *= scale_y
        return bboxes, labels, scores

    def _get_region_boxes(self, output, img_W, img_H):
        conf_thresh = 0.1
        B, C, H, W = output.shape

        assert C == 19 + self.n_class

        # (B, H, W)
        grid_x = np.linspace(0, W-1,W)[None].repeat(H, 0)[None].repeat(B, 0)
        grid_y = np.linspace(0, H-1, H)[None].repeat(W, 0).T[None].repeat(B, 0)

        x0 = F.sigmoid(output[:, 0]).data + grid_x
        y0 = F.sigmoid(output[:, 1]).data + grid_y
        xs = output[:, 2:18:2] + grid_x[:, None]
        ys = output[:, 3:18:2] + grid_y[:, None]

        det_confs = F.sigmoid(output[:, 18]).data
        cls_conf = F.softmax(output[:, 19:19 + self.n_class]).data

        cls_max_ids = self.xp.argmax(cls_conf, axis=1)
        cls_max_confs = self.xp.max(cls_conf, axis=1)

        bboxes = []
        labels = []
        scores = []
        max_conf = -100000
        for b in range(B):
            bbox = []
            label = []
            score = []

            max_conf = -1
            for cy in range(H):
                for cx in range(W):
                    # only_objectness == True
                    det_conf = det_confs[b, cy, cx]
                    conf = det_conf

                    if conf > max_conf:
                        max_conf = conf
                        max_ind = (cy, cx)

                    if conf > conf_thresh:
                        bx0 = (x0[b, cy, cx] / W) * img_W
                        by0 = (y0[b, cy, cx] / H) * img_H
                        bxs = (xs[b, :, cy, cx] / W) * img_W
                        bys = (ys[b, :, cy, cx] / H) * img_H

                        cls_max_conf = cls_max_confs[b, cy, cx]
                        cls_max_id = cls_max_ids[b, cy, cx]

                        bb = np.zeros((9, 2), dtype=np.float32)
                        bb[0] = [bx0, by0]
                        bb[1:, 0] = bxs
                        bb[1:, 1] = bys
                        # box[18] = det_conf
                        # box[19] = cls_max_conf
                        # box[20] = cls_max_id

                        # TODO: logic when only_objectness == False
                        bbox.append(bb)
                        label.append(cls_max_id)
                        score.append(det_conf)

            if len(bbox) == 0:
                raise ValueError
            bboxes.append(self.xp.array(bbox, dtype=np.float32))
            labels.append(self.xp.array(label, dtype=np.int32))
            scores.append(self.xp.array(score, dtype=np.float32))
        return bboxes, labels, scores



if __name__ == '__main__':
    img = np.load('img.npy')[0] * 255
    _, H, W = img.shape

    output = np.load('input.npy')
    model = SSPYOLOv2()
    all_boxes, _, _ = model._get_region_boxes(output, W, H)

    import pickle
    with open('a.pkl', 'rb') as f:
        all_boxes_gt = pickle.load(f)

    from chainercv.visualizations import vis_image
    import matplotlib.pyplot as plt
    img = np.load('img.npy')[0] * 255
    _, H, W = img.shape

    bbox = all_boxes[0][0]
    vis_image(img)
    bbox = np.array(bbox[:18]).reshape(9, 2)
    for i in range(9):
        plt.scatter(bbox[i][0], bbox[i][1]) 
    plt.show()



    # print(all_boxes[0][0])
    # print(all_boxes_gt[0][0])
