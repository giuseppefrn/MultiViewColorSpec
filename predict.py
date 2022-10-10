import argparse
import tensorflow as tf
import numpy as np
import os

from matplotlib import pyplot as plt
from skimage import color

from models.model import build_model
from utils.metrics import delta_e
from utils.data_loader import load_and_preprocess_rgb, load_and_preprocess_rgbd

def predict(opt):

    model = build_model(N_view=opt.n_views, mode=opt.mode)
    print('Model loaded')

    model.load_weights(opt.weights)
    print('Weights loaded')

    #load images from the given dir
    #apply preprocessing
    xs = []
    if opt.mode == 'RGB':
        xs = load_and_preprocess_rgb(opt.data_dir, opt.n_views)
    else:
        xs = load_and_preprocess_rgbd(opt.data_dir, opt.n_views)

    xs = np.expand_dims(xs, 0)


    #predict
    preds = model([xs[:,i] for i in range(opt.n_views)], training=False) #use the random func?

    preds = preds.numpy()

    np.save(os.path.join(opt.output_dir, 'preds'), preds)

    print('Predictions:')
    print(preds[0])
    print('Results in', os.path.join(opt.output_dir, 'preds'))

    k = np.ones(shape=(30,30,3))
    img = k * preds[0]
    img_rgb = color.lab2rgb(img)
    plt.imshow(img_rgb)
    plt.savefig(os.path.join(opt.output_dir, 'predicted.png'))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to the data', required=True)
    parser.add_argument('--output_dir', type=str, default='experiments', help='output directory pathname')
    parser.add_argument('--n_views', type=int, default=16, choices=[16,8,4,3,2,1], help='number of views to use')
    parser.add_argument('--weights', type=str, help='path to the model weights', required=True)
    parser.add_argument('--mode', type=str, default='RGBD', help='RGBD or RGB mode', choices=['RGBD', 'RGB'])

    opt = parser.parse_args()

    predict(opt)