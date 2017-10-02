from shared import *

class pix():
    axis = 'z'
    dataset_name = 'Combined7c_ne'
    input_c_dim = 4
    output_c_dim = 3
    speed_factor = 1
    batch_size = 10
    
    # size of first generator convolution
    gf_dim = 128
    
    # size of first discriminator convolution
    df_dim = 128
    
    # learning rate
    lr = 0.0002 # 0.0002
    
    # phase is train, test or contest
    phase = 'train'
    
var = pix()
init(var)
run(var)
