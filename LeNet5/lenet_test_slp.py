import os
import cv2
import sophon.sail as sail
import numpy as np
import time

def preprocess(img_file):
    img = cv2.imread(filename=img_file, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    img -= 127.5
    img *= 1 / 127.5
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs):
    #outputs = outputs.detach().numpy()
    outputs = list(outputs.values())[0]
    pred_idx = outputs.argmax()
    pred_conf = outputs.max()
    return pred_idx, pred_conf

def infer(input_path, model_path):
    # 1.load model锛宲lease fill in the code
    model = sail.Engine(model_path, 0, sail.IOMode.SYSIO)
    graph_name = model.get_graph_names()[0]  # get net name
    input_name = model.get_input_names(graph_name)[0]  # get input names
    
    if os.path.isfile(input_path):
        # 2.preprocess锛宲lease fill in the code
        input_array = preprocess(input_path)
        # 3.inference锛宲lease fill in the code
        input_data = {input_name: input_array}
        outputs = model.process(graph_name, input_data)
        # 4.postprocess
        pred_idx, pred_conf = postprocess(outputs)
        print("{}: pred_idx={}, pred_conf={:.2f}".format(input_path, pred_idx, pred_conf))
        return pred_idx, pred_conf
    else:
        correct_num = 0.0
        total_num = 0.0
        preprocess_time, inference_time, postprocess_time = 0, 0, 0
        # please fill in the code
        for img_root, _, img_names in os.walk(input_path):
            for img_name in img_names:
                time1 = time.time()
                input_array = preprocess(os.path.join(img_root, img_name))
                time2 = time.time()
                #outputs = model(input_tensor)
                input_data = {input_name: input_array}
                outputs = model.process(graph_name, input_data)
                time3 = time.time()
                pred_idx, pred_conf = postprocess(outputs)
                time4 = time.time()
                preprocess_time += time2 - time1
                inference_time += time3 - time2
                postprocess_time += time4 - time3
                label = os.path.split(img_root)[-1]
                print("{}: label={}, pred_idx={}, pred_conf={:.2f}".format(img_name, label, pred_idx, pred_conf))
                total_num += 1
                if pred_idx == int(label):
                    correct_num += 1

        print("acc = {}/{} = {:.2f}%".format(correct_num, total_num, correct_num / total_num * 100))
        print("preprocess_time={:.2f}ms, inference_time={:.2f}ms, postprocess_time={:.2f}ms". \
              format(postprocess_time/total_num*1000, inference_time/total_num*1000, postprocess_time/total_num*1000))
        return -99, correct_num / total_num * 100

if __name__ == '__main__':
    model_path = 'compilation.bmodel'
    input_path = 'test'
    infer(input_path, model_path)


