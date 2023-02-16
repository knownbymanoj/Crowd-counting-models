import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM


def gaussian_filter_density(img,points):
    '''


    '''
    img_shape=[480,640]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    root = 'M:\\VisionandCognition\\MCNN\\MCNN-pytorch'
    
    # now generate the crowd_countings's ground truth
    images = os.path.join(root,'mall_dataset_online','images')
    path_sets = [images]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for i,img_path in enumerate(img_paths):
        print(img_path)
        mat = io.loadmat(os.path.join(root,'mall_dataset_online','mall_gt.mat'))
        img= plt.imread(img_path)#480*640
        k = np.zeros((480,640))
        points = mat['frame'][0][i][0][0][0]
        k = gaussian_filter_density(img,points)
        # plt.imshow(k,cmap=CM.jet)
        # save density_map to disk
        np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth_density_maps'), k)
    

    #now see a sample from ShanghaiA
    plt.imshow(Image.open(img_paths[0]))
    
    gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth_density_maps'))
    plt.imshow(gt_file,cmap=CM.jet)
    
    print(np.sum(gt_file))# don't mind this slight variation

    