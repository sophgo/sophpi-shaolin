import cv2
import time
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


class ReconNet():
    def __init__(self, pb_path):
        self.pb_path = pb_path
        self.config = tf.ConfigProto()
        self.init_model()

    def init_model(self):
        tf.Graph().as_default()
        self.output_graph_def = tf.GraphDef()
        with open(self.pb_path, 'rb') as f:
            self.output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(
                self.output_graph_def,
                input_map=None,
                return_elements=None,
                name=None,
                op_dict=None,
                producer_op_list=None)
        self.sess = tf.Session(config=self.config)  # 模型输入节点
        # self.input = self.sess.graph.get_tensor_by_name('import/x:0')  # 模型输出节点
        self.input = self.sess.graph.get_tensor_by_name('x:0')
        # self.output = self.sess.graph.get_tensor_by_name('import/recon_block/conv2d_5/LeakyRelu:0')
        self.output = self.sess.graph.get_tensor_by_name('recon_block/conv2d_5/LeakyRelu:0')

    def reconnet(self, observed):
        out = self.sess.run(self.output, feed_dict={self.input: observed})
        out = np.squeeze(out)
        out = np.uint8(out)

        return out


def get_img(img):
    test_image = cv2.imread(img)
    test_image_lum = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCR_CB)
    test_image_lum = test_image_lum[:, :, 0]
    # test_image_lum = cv2.normalize(test_image_lum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    test_gray = cv2.imread(img, 0)
    return test_image_lum, test_gray


def main():
    test_image_lum, test_gray = get_img(img_path)

    filter_size = 33
    stride = 14
    (h, w) = test_image_lum.shape
    h_iters = ((h - filter_size) // stride) + 1
    w_iters = ((w - filter_size) // stride) + 1

    model = ReconNet(pb_path=pb_path)
    recon_img = np.zeros((h, w))
    start = time.time()
    for i in range(h_iters):
        for j in range(w_iters):
            observed = np.expand_dims(
                test_image_lum[stride * i:filter_size + stride * i, stride * j:filter_size + stride * j], axis=0)

            output = model.reconnet(observed)
            recon_img[stride * i:filter_size + stride * i, stride * j:filter_size + stride * j] = output
    print('reconstruction img saved as recon_img.jpg')
    print("cost time: {:.5f} s".format(time.time() - start))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(recon_img, cmap='gray')
    plt.title('reconstruction_img')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(test_gray, cmap='gray')
    plt.title('test_img')
    plt.axis('off')
    plt.show()
    cv2.imwrite('recon_img.jpg', recon_img)


if __name__ == '__main__':
    pb_path = './reconnet_model.pb'
    img_path = './ReconNet_SE5/test.jpg'
    main()
