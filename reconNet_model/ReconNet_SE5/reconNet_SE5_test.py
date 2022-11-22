import cv2
import time
import argparse
import numpy as np
import sophon.sail as sail


def get_img(img):
    test_image = cv2.imread(img)
    test_image_lum = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCR_CB)
    test_image_lum = test_image_lum[:, :, 0]
    # test_image_lum = cv2.normalize(test_image_lum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    test_gray = cv2.imread(img, 0)
    return test_image_lum, test_gray


def main():
    test_image_lum, test_gray = get_img(ARGS.input)

    filter_size = 33
    stride = 14
    (h, w) = test_image_lum.shape
    h_iters = ((h - filter_size) // stride) + 1
    w_iters = ((w - filter_size) // stride) + 1

    net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
    graph_name = net.get_graph_names()[0]
    input_names = net.get_input_names(graph_name)

    recon_img = np.zeros((h, w))
    start_time = time.time()  # 开始时间
    for i in range(h_iters):
        for j in range(w_iters):
            observed = np.expand_dims(
                test_image_lum[stride * i:filter_size + stride * i, stride * j:filter_size + stride * j], axis=0)
            input_data = {input_names[0]: np.array(observed, dtype=np.float32)}

            output = net.process(graph_name, input_data)
            recon_img[stride * i:filter_size + stride * i, stride * j:filter_size + stride * j] = output['recon_block/conv2d_5/LeakyRelu'].squeeze(-1)
    print("-" * 66)
    print('reconstruction img saved as recon_img.jpg')
    print("cost time: {:.5f} s".format(time.time() - start_time))
    print("-" * 66)
    cv2.imwrite('recon_img.jpg', recon_img)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for sail py test')
    PARSER.add_argument('--bmodel', default='./reconNet_bmodel/compilation.bmodel')
    PARSER.add_argument('--input', default='./test.jpg')
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    ARGS = PARSER.parse_args()
    main()