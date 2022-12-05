import sophon.sail as sail 
import numpy as np 


class EngineOV:
    
    def __init__(self, model_path="",device_id=0) :
        # net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
        self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        self.graph_name = self.model.get_graph_names()[0]
        self.input_name = self.model.get_input_names(self.graph_name)[0]
        self.input_shape= self.model.get_input_shape(self.graph_name, self.input_name)
        self.output_name= self.model.get_output_names(self.graph_name)[0]
        self.output_shape= self.model.get_output_shape(self.graph_name, self.output_name)
        print("input_name={}, input_shape={}, output_name={}, output_shape={}".format(self.input_name,self.input_shape,self.output_name,self.output_shape))
        
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
    def __check(self,args:dict):
        # check input: input should only contain input_name and input_data should be numpy array with shape of input_shape
        if self.input_name not in args.keys():
            raise Exception("input_name not in args")
        if not isinstance(args[self.input_name],np.ndarray):
            raise Exception("input_data should be numpy array")
        # if args[self.input_name].shape != self.input_shape:
        #     raise Exception("input_data shape should be {}".format(self.input_shape))
        
    
    def __call__(self, args):
        # self.__check(args)
        output = self.model.process(self.graph_name, args)
        return output[self.output_name]
