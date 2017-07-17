from shared import *

IMG_SIZE = 256  # 128

class pix_env():
    axis = 'z'  # 'x'
    dataset_name = 'Png-Combined_ao'  # 'Png-HGG7b'
    input_c_dim = 4
    output_c_dim = 3
    speed_factor = 256 // IMG_SIZE
    batch_size = 1
    gf_dim = 128
    df_dim = 128
    lr = 0.00005 # 0.0002
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

def store_input(pix, dir_='val', name='Brats17_2013_10_1_combined.nx.77.png', only_output=False):
    # load full image
    path = os.path.join(dir_, name)
    sample = load_data(path, pix.image_size, pix.input_c_dim, pix.output_c_dim if not only_output else 0)
    # np.set_printoptions(threshold=np.inf)
    if only_output:
        scipy.misc.imsave(os.path.join('output', 'src_' + name), sample[:,:,2])
    else:
        if IMG_SIZE*IMG_SIZE + sample[:,:,0].sum() < 100:  # black pixels are -1
            # Ignore brain slices having no tumor in it
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
img1 = np.array([[color1] * IMG_SIZE] * IMG_SIZE)
img2 = np.array([[color2] * IMG_SIZE] * IMG_SIZE)
img3 = np.array([[color3] * IMG_SIZE] * IMG_SIZE)
img_empty = np.ones((IMG_SIZE, IMG_SIZE, 3))
color_images = np.array([img1, img2, img3])

def store_output(pix, result, name='Brats17_2013_10_1_combined.nx.77.png'):
    filler = -np.ones((IMG_SIZE, IMG_SIZE))
    for i in range(3):
        region_img = result[:, :, i]
        alphas = (region_img + 1) / 2
        if alphas.sum() < 1:
            rgba_img = img_empty
        else:
            rgba_img = np.array([
                color_images[i,:,:,0],
                color_images[i,:,:,1],
                color_images[i,:,:,2],
                region_img]).transpose((1, 2, 0))
        scipy.misc.imsave(os.path.join('output', str(i) +'_' + name), rgba_img)
    # raise ValueError('STOP')
    # combined = np.concatenate([result[:,:,0], result[:,:,1], result[:,:,2]])
    # scipy.misc.imsave(os.path.join('output', name), result[:,:,:3])

def execute(pix, img_path, with_gt=True):
    generated = []
    print('Executing on {}'.format(img_path))
    dir_, name = os.path.split(img_path)
    if not store_input(pix, dir_, name, not with_gt):
        print('Failed for {}'.format(name))
        return generated
    generated.append(os.path.join('output', 'src_' + name))
    img = read_input(pix, 'input' if with_gt else 'uploads', name)
    result = run(pix, img, name)
    store_output(pix, result, name)
    generated.append(os.path.join('output', '0_' + name))
    generated.append(os.path.join('output', '1_' + name))
    generated.append(os.path.join('output', '2_' + name))
    return generated


if __name__ == '__main__':
    pix = build()
    for img_path in glob('val/*.n{}.*.png'.format(pix.axis)):
        execute(pix, img_path)
