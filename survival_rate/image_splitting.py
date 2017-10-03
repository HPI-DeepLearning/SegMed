from PIL import Image
import glob
import os
from tqdm import tqdm

def long_slice(image_path, out_name, outdir, slice_size):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size
    upper = 0
    left = 0
    slices = height//slice_size

    count = 1
    for slice in range(slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count == slices:
            lower = height
        else:
            lower = int(count * slice_size)

        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
        #save the slice
        working_slice.save(os.path.join(outdir, out_name + "_" + str(count)+".png"))
        count +=1

if __name__ == '__main__':
    long_files = glob.glob("data/raw/*.png")
    for f in tqdm(long_files):
        long_slice(f, f.replace("raw", "processed").replace(".png", ""), os.getcwd(), 128)
