from shared import *

class pix():
    axis = 'x'
    dataset_name = 'Png-HGG7b'
    input_c_dim = 4
    output_c_dim = 3
    speed_factor = 1
    batch_size = 24
    gf_dim = 128
    df_dim = 128
    lr = 0.0002
	# phase is train, test or contest
    phase = 'train'
    
var = pix()
init(var)
run(var)
