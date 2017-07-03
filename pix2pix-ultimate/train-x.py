from shared import *

class pix():
    axis = 'x'
    dataset_name = 'Png-HGG7b'
    input_c_dim = 4
    output_c_dim = 3
    speed_factor = 2
    batch_size = 16
    gf_dim = 64
    df_dim = 64
    phase = 'test'
    
var = pix()
init(var)
run(var)
