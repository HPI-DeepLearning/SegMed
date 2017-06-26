from shared import *

class pix():
    axis = 'x'
    dataset_name = 'Combined5'
    input_c_dim = 4
    output_c_dim = 1
    speed_factor = 2
    batch_size = 32
    gf_dim = 64
    df_dim = 64
    phase = 'train'
    
var = pix()
init(var)
run(var)