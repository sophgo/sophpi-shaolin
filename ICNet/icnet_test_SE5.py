import cv2
import time
import argparse
import numpy as np
import sophon.sail as sail


def icnet_preprocess(image, infer_size=(1024, 2048, 3)):
    """ICNEt img preprocess
    resize image with shape(1024,2048,3)
    """
    if image.shape != infer_size:
        image = cv2.resize(image, (infer_size[1], infersize[0]))
    return image


def icnet_postprocess(output, detected_size, num_classes, label_colours):
    """Parse the yolov3 output

    To be improve
    """
    label_colours = np.array(label_colours, np.float32)
    detected_w, detected_h = detected_size[1], detected_size[0]
    output = np.transpose(output, (0, 2, 1, 3)).squeeze()  # [1,255,511,19] -> [255,511,19]
    # [255,511,19] -> [1024,2048,19]
    output_up = cv2.resize(output, (detected_w, detected_h), interpolation=cv2.INTER_LINEAR)
    output_label = np.argmax(output_up, axis=2)
    onehot_output_label = np.eye(num_classes)[output_label]
    result = np.matmul(onehot_output_label, label_colours)
    return result


def main():
    num_classes = 19
    infer_size = (1024, 2048, 3)  # input size about ICNet
    detected_size = (1024, 2048)

    label_colours = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                     # 0 = road, 1 = sidewalk, 2 = building
        , [102, 102, 156], [190, 153, 153], [153, 153, 153]
                     # 3 = wall, 4 = fence, 5 = pole
        , [250, 170, 29], [219, 219, 0], [106, 142, 35]
                     # 6 = traffic light, 7 = traffic sign, 8 = vegetation
        , [152, 250, 152], [69, 129, 180], [219, 19, 60]
                     # 9 = terrain, 10 = sky, 11 = person
        , [255, 0, 0], [0, 0, 142], [0, 0, 69]
                     # 12 = rider, 13 = car, 14 = truck
        , [0, 60, 100], [0, 79, 100], [0, 0, 230], [119, 10, 32]]
    # 15 = bus, 16 = train, 17 = motocycle, 18 = bicycle

    # load img
    img = cv2.imread(ARGS.input)
    # preprocess  ICNet
    data = icnet_preprocess(img, infer_size)
    print('data.shape=', data.shape)

    # sail core (inference)
    net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
    graph_name = net.get_graph_names()[0]  # get net name
    input_names = net.get_input_names(graph_name)  # get input names

    # original script
    output_names = net.get_output_names(graph_name)  # get input names
    input_data = {input_names[0]: np.array(data, dtype=np.float32)}
    start = time.time()
    output = net.process(graph_name, input_data)
    end = time.time()

    # postprocess ICNet
    result = icnet_postprocess(output[output_names[0]], detected_size, num_classes, label_colours)
    overlap_result = 0.5 * img + 0.5 * result
    cv2.imwrite('result.jpg', result)
    cv2.imwrite('overlap_result.jpg', overlap_result)
    print("-" * 66)
    print('saved as result.jpg, overlap_result.jpg')
    print("cost time: {} s".format(end - start))
    print("-" * 66)


# main
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for sail py test')
    PARSER.add_argument('--bmodel', default='compilation.bmodel')
    PARSER.add_argument('--input', default='data/cityscapes1.png')
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    ARGS = PARSER.parse_args()
    main()
