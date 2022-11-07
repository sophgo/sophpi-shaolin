import bmnetu
## compile int8 model
bmnetu.compile(
    model = "../data/int8model/flasr_a_fp32_bmnett_deploy_int8_unique_top.prototxt", ## Necessary
    weight = "../data/int8model/flasr_a_fp32_bmnett.int8umodel", ## Necessary
    outdir = "../data/int8model", ## Necessary
    prec = "INT8", ## optional, if not set, default use FP32
    # shapes = [[1,240,432,1], [1,480,864,2]], ## optional, if not set, default use shape in prototxt
   # net_name = "name", ## optional, if not set, default use the network name␣ ,→in prototxt
    opt = 2, ## optional, if not set, default equal to 2
    dyn = False, ## optional, if not set, default equal to False
    cmp = True ## optional, if not set, default equal to True
)