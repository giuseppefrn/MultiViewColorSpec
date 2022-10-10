#Generate dataset from RGB-D images and LAB given a GT csv
import argparse
import re
import tensorflow as tf
import numpy as np
import os
import pandas as pd

from data_loader import load_and_preprocess_rgbd

#get true Lab from csv, in order to use yours change create a similar function
def get_lab(color_name, gt_df, light='D65'):
    """
    V01 - use only the color name and the light to get the lab value. We have 1 Lab for each cube
    Get the true L,a,b value
    Args:
    - color_name: color code used to find the lab value must be Color_n n = 0,1,2,3...
    - light: must be "D65" or "840"
    """

    ill = '_' + light

    tmp = gt_df.loc[gt_df['name'] == color_name, ['Status', 'L*' + ill, 'a*' + ill, 'b*' + ill]]
    tmp = tmp.loc[tmp['Status'] == 'SCE/100']

    L = tmp.loc[:, 'L*' + ill].mean() #mean of 3 values
    a = tmp.loc[:, 'a*' + ill].mean()
    b = tmp.loc[:, 'b*' + ill].mean()

    return [L, a, b]

def create_multiview_rgbd_dataset(opt):
    df = np.array([],dtype=[('shape', 'U4'), ('color', 'U10'),('subcolor', np.int8), ('light', 'U4'), ('data', np.float16, (opt.n_views, 128, 128, 4)),('LAB', np.float16, (3,))])
    
    #we assume just one shape and no subcolors
    # The folder shoud have this schema
    #   - Illuminant
    #     - shape (cube)
    #       - color_0
    #           - RGB + D images
    #       - color_1
    #       - ...
    shapes_list = os.listdir(opt.data_dir)

    final_output = os.path.join(opt.output_dir, opt.illuminant)

    gt_df = pd.read_csv(opt.gt)

    for s in shapes_list:
        colors_list = os.listdir(os.path.join(opt.data_dir, s))
        for c in colors_list:
            #inside the color folder we expect N RGB and N Depth images

            c_path = os.listdir(os.path.join(opt.data_dir, s, c))

            #load and preprocess all rgbd images in the folder
            rgbd = load_and_preprocess_rgbd(os.path.join(opt.data_dir, s, c),opt.n_views)

            #get LAB value from gt
            #default illuminant is D65
            lab = get_lab(c, gt_df)
            lab = np.ones(shape=(opt.n_views, 3)) * lab

            #merge rgbd, lab, light etc 
            #default subcolor is 0 here, we keep it just to retro comp
            data = np.array([([s]*16, [c] * 16, [0] * 16, [opt.illuminant]* 16, rgbd,lab)], #replace 1,2,3 with L* a* b*
                  dtype=[('shape', 'U4',(16,)),('color', 'U10',(16,)), ('subcolor', np.int8,(16,)) ,('light', 'U4', (16,)), ('data', np.float16, (16, 128, 128, 4)), ('LAB', np.float16, (16,3))])
        
        
            #append to df
            # if df.shape[0] == 0:
            #         df = data
            # else:
            #     df = np.append(df, data, axis = 0)
            
            os.makedirs(os.path.join(final_output,s), exist_ok=True)
            np.save(os.path.join(final_output,s,c),data)

if __name__ == '__main__':

    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--illuminant', type=str, default='D65', help='illuminant to use [D65, F11, F4, ...]')
    parser.add_argument('--data_dir', type=str, required=True, help='path to the images dataset')
    parser.add_argument('--output_dir', type=str, default='multiviews', help='output directory pathname where save the dataset')
    parser.add_argument('--n_views', type=int, default=16, choices=[16,8,4,3,2,1], help='number of views to use, 1 for single view model')
    parser.add_argument('--gt', type=str, help='gt.csv pathname', required=True)
    

    opt = parser.parse_args()


    create_multiview_rgbd_dataset(opt)
