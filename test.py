import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import os
import csv 
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import numpy as np


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    print('started mae calculation')
    device=torch.device("cpu")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            img_name,img,gt_dmap = data 
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()- gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap,img_name

    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    print('started estimating density map')
    device=torch.device("cpu")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    with open('test_mall_with_part_B.csv','w') as fileObj:
        writerObj = csv.writer(fileObj)
        writerObj.writerow(['location of image','Original no of persons in Density Map','Predicted no of persons in Density Map'])
        for i,data in enumerate(dataloader):
                img_name,img,gt_dmap = data 
            #if i==index:
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)
                # forward propagation
                et_dmap=mcnn(img).detach()
                et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
                writerObj.writerow([img_name,gt_dmap.sum().item(),et_dmap.sum()])
                print(img_name[0])
                #if img_name[0] == 'seq_001607.jpg':
                #    print('DONE')
                #    np.save('dmap_mall_1607.npy', et_dmap)
                #print(et_dmap.shape)
                #break


if __name__=="__main__":
    print('started')
    torch.backends.cudnn.enabled=False
    root_dir = 'M:\\VisionandCognition\\MCNN\\MCNN-pytorch'
    #img_root= os.path.join(root_dir,'ShanghaiTech','part_B','train_data','images')
    #gt_dmap_root=os.path.join(root_dir,'ShanghaiTech','part_B','train_data','ground-truth')
    #img_root= os.path.join(root_dir,'mall_dataset_online','images')
    #gt_dmap_root=os.path.join(root_dir,'mall_dataset_online','ground_truth_density_maps')
    img_root= os.path.join(root_dir,'mall_dataset_online','test_images')
    gt_dmap_root=os.path.join(root_dir,'mall_dataset_online','test_gt')
    model_param_path=os.path.join(root_dir,'checkpoints','epoch_348.param')
    
    cal_mae(img_root,gt_dmap_root,model_param_path)

    #estimate_density_map(img_root,gt_dmap_root,model_param_path,3)
