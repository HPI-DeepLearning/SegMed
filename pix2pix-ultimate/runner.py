from shared import *

class pix_env():
    axis = 'x'
    dataset_name = 'Png-HGG7b'
    input_c_dim = 4
    output_c_dim = 3
    speed_factor = 2
    batch_size = 16
    gf_dim = 64
    df_dim = 64
    phase = 'test'

def build():
    pix = pix_env()
    init(pix)
    pix.sess = tf.Session()
    build_model(pix)
    init_op = tf.global_variables_initializer()
    pix.sess.run(init_op)
    if load(pix, pix.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        raise ValueError(" [!] Load failed...")
    return pix

def run(pix, input_image, name):
    print('Predict {}'.format(name))
    sample_image = np.array([input_image] * pix.batch_size)
    samples = pix.sess.run(
        pix.fake_B_sample,
        feed_dict={pix.real_data: sample_image}
    )
    return samples[0]

def store_input(pix, dir_='val', name='Brats17_2013_10_1_combined.nx.77.png'):
    # load full image
    path = os.path.join(dir_, name)
    sample = load_data(path, pix.image_size, pix.input_c_dim, pix.output_c_dim)
    sample_images = np.array(sample).astype(np.float32)
    np.set_printoptions(threshold=np.inf)
    if 128*128 + sample[:,:,0].sum() < 100:
        return False
    o = np.concatenate([sample[:,:,3], sample[:,:,4], sample[:,:,5], sample[:,:,6]], axis=1)
    scipy.misc.imsave(os.path.join('input', name), o)
    scipy.misc.imsave(os.path.join('output', 'src_' + name), sample[:,:,5])
    return True

def read_input(pix, dir_='input', name='Brats17_2013_10_1_combined.nx.77.png'):
    path = os.path.join(dir_, name)
    sample = load_data(path, pix.image_size, pix.input_c_dim, 0)
    filled = np.concatenate([np.zeros((pix.image_size, pix.image_size, pix.output_c_dim)), sample], axis=2)
    return filled

color1 = [0.96, 0.76, 0.2]
color2 = [0.82, 0.36, 0.18]
color3 = [0.16, 0.56, 0.78]
img1 = np.array([[color1] * 128] * 128)
img2 = np.array([[color2] * 128] * 128)
img3 = np.array([[color3] * 128] * 128)
color_images = np.array([img1, img2, img3])
def store_output(pix, result, name='Brats17_2013_10_1_combined.nx.77.png'):
    filler = -np.ones((128, 128))
    for i in range(3):
        region_img = result[:, :, i]
        alphas = (region_img + 1) / 2
        rgba_img = np.array([
            color_images[i,:,:,0],
            color_images[i,:,:,1],
            color_images[i,:,:,2],
            region_img]).transpose((1, 2, 0))
        scipy.misc.imsave(os.path.join('output', str(i) +'_' + name), rgba_img)
    # raise ValueError('STOP')
    # combined = np.concatenate([result[:,:,0], result[:,:,1], result[:,:,2]])
    # scipy.misc.imsave(os.path.join('output', name), result[:,:,:3])

if __name__ == '__main__':
    pix = build()
    for img_path in glob('val/*.n{}.*.png'.format(pix.axis)):
        dir_, name = os.path.split(img_path)
        # name='Brats17_2013_10_1_combined.nx.77.png'
        if not store_input(pix, dir_, name):
            print('Skipping {}'.format(name))
            continue
        img = read_input(pix, 'input', name)
        result = run(pix, img, name)
        store_output(pix, result, name)
