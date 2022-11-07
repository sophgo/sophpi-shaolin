import sophon.sail as sail
import numpy as np

class Resnet18:
    def __init__(self, bmodel_path, tpu_id):
        # load bmodel
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.handle = self.net.get_handle()

        # get model info
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]


    def preprocess(self):
        raise NotImplementedError


    def predict(self, img):

        input = {self.input_name: np.array(img, dtype=np.float32)}
        output = self.net.process(self.graph_name, input)
        return list(output.values())


    def postprocess(self):
        raise NotImplementedError
    

    def inference(self, img):
        output = self.predict(img)
        return output