import bmnett
## compile fp32 model
bmnett.compile(
  model = "AnimeGAN_dynamic.pb",     ## Necessary
  outdir = "./compilation1684",                    ## Necessary
  target = "BM1684",                 ## Necessary
  shapes = [[1,700,1024,3]],     ## Necessary
  net_name = "animate",                 ## Necessary
  input_names=["test"],    ## Necessary, when .h5 use None
  output_names=["generator/G_MODEL/Tanh"], ## Necessary, when .h5 use None
  opt = 2,                           ## optional, if not set, default equal to 1
  dyn = False,                       ## optional, if not set, default equal to False
  cmp = False,                        ## optional, if not set, default equal to True
  enable_profile = False              ## optional, if not set, default equal to False
)