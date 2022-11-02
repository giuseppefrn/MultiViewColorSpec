import argparse
import numpy as np
import pandas as pd
import os

from utils.data_loader import get_data_illuminant_data, get_random_k_view, split_on_value, get_data_from_dir
from utils.metrics import delta_e, delta_e_2000
from models.model import build_model
from utils.plot_utils import plot_hist_pred_delta_e, plot_boxplot_pred_delta_e

def predict_on_df(model,df,opt):

    columns = ['shape', 'color', 'subcolor', 'light', 'LAB', 'pred_LAB', 'DELTA']
    result_df = pd.DataFrame(columns=columns)

    for i in range(len(df)):
        color = df[i]['color'][0]
        subcolor = df[i]['subcolor'][0]
        LAB = np.expand_dims(df[i]['LAB'][0],0)
        mesh = df[i]['shape'][0]
        light = df[i]['light'][0]

        data = df[i]['data']

        if opt.mode == 'RGB':
            data = data[:,:,:,:3]
        
        data = np.expand_dims(data, 0)

        #ordered fixed views 
        # pred_lab = model([data[:, i] for i in views_dict_idxs[n_views]])

        #random views
        pred_lab = model([data[:, i] for i in get_random_k_view(opt.n_views)])
        
        delta = delta_e_2000(pred_lab, LAB)

        
        result_df = result_df.append( {'shape': mesh, 'color':color, 'subcolor':subcolor,'light':light, 'LAB':LAB[0], 'pred_LAB':pred_lab.numpy()[0], 'DELTA': delta}
            ,ignore_index=True)
    return result_df

def predict_from_folder_data(opt, model, final_output_dir):
    #load data
    train_dataset = get_data_from_dir(os.path.join(opt.data_dir, 'train'))
    valid_dataset = get_data_from_dir(os.path.join(opt.data_dir, 'valid'))
    test_dataset = get_data_from_dir(os.path.join(opt.data_dir, 'test'))

    #predict
    result_train = predict_on_df(model, train_dataset, opt)
    result_valid = predict_on_df(model, valid_dataset, opt)
    result_test = predict_on_df(model, test_dataset, opt)

    #save results
    os.makedirs(final_output_dir, exist_ok=True)
    result_train.to_pickle(os.path.join(final_output_dir,'predictions-train.pkl'))
    result_valid.to_pickle(os.path.join(final_output_dir,'predictions-valid.pkl'))
    result_test.to_pickle(os.path.join(final_output_dir,'predictions-test.pkl'))
    print('Reults saved')

    return result_train, result_valid, result_test

def predict_illuminant_data(opt, model, final_output_dir):
    df = get_data_illuminant_data(opt.data_dir, opt.illuminant)

    # columns = ['mesh', 'color', 'subcolor', 'LAB', 'pred_LAB', 'DELTA']
    # result_df = pd.DataFrame(columns=columns)

    # # views_dict_idxs = {16: range(16), 8:range(0,16,2), 4:[12, 14, 0, 2], 3:[12, 0, 4], 2:[0, 4]}

    # for i in range(len(df)):
    #     color = df[i]['color'][0]
    #     subcolor = df[i]['subcolor'][0]
    #     LAB = np.expand_dims(df[i]['LAB'][0],0)
    #     mesh = df[i]['shape'][0]

    #     data = df[i]['data']

    #     if opt.mode == 'RGB':
    #         data = data[:,:,:,:3]
        
    #     data = np.expand_dims(data, 0)

    #     #ordered fixed views 
    #     # pred_lab = model([data[:, i] for i in views_dict_idxs[n_views]])

    #     #random views
    #     pred_lab = model([data[:, i] for i in get_random_k_view(opt.n_views)])
        
    #     delta = delta_e(pred_lab, LAB)

        
    #     result_df = result_df.append( {'mesh': mesh, 'color':color, 'subcolor':subcolor, 'LAB':LAB[0], 'pred_LAB':pred_lab.numpy()[0], 'DELTA': delta.numpy()}
    #         ,ignore_index=True)

    result_df = predict_on_df(model, df, opt)

    
    # result_df.to_pickle(os.path.join(final_output_dir,'predictions.pkl'))
    # print('Reults saved')

    try:
        train_idxs = np.load(os.path.join(final_output_dir, 'train_indexs.npy'))
        valid_idxs = np.load(os.path.join(final_output_dir, 'validation_indexs.npy'))
        
    except:
        print('Error. Cannot load train and validation idxs')
        return -1
    
    result_df = result_df.to_numpy()
    return result_df, train_idxs, valid_idxs

def evaluate(opt):
    weights = opt.weights
    n_views = opt.n_views

    model = build_model(N_view=n_views, mode=opt.mode)

    print('Model loaded')

    #load weights
    model.load_weights(weights)

    print('Weights loaded')
    print(model.summary())

    final_output_dir = opt.output_dir

    columns = ['shape', 'color', 'subcolor', 'light', 'LAB', 'pred_LAB', 'DELTA']

    #switch here
    if opt.test_on == 'folder_split':
        results_train, results_valid, results_test = predict_from_folder_data(opt, model, final_output_dir)
    else:
        result_df, train_idxs, valid_idxs = predict_illuminant_data(opt, model, final_output_dir)
        columns.index(opt.test_on)
        result_df, results_test = split_on_value(result_df, columns.index(opt.test_on) , opt.value, 1)

        # take train and valid
        results_train = result_df[train_idxs]
        results_valid = result_df[valid_idxs]

        os.makedirs(final_output_dir, exist_ok=True)

        results_train = pd.DataFrame(data = results_train, 
                  index = np.arange(len(results_train)), 
                  columns = columns)
        
        results_valid = pd.DataFrame(data = results_valid, 
                  index = np.arange(len(results_valid)), 
                  columns = columns)
        
        results_test = pd.DataFrame(data = results_test, 
                  index = np.arange(len(results_test)), 
                  columns = columns)

        results_test.to_pickle(os.path.join(final_output_dir,'predictions-test.pkl'))
        results_train.to_pickle(os.path.join(final_output_dir,'predictions-train.pkl'))
        results_valid.to_pickle(os.path.join(final_output_dir,'predictions-valid.pkl'))

    
    # PLOT SECTION
    
    # results_test = result_df[np.unique(np.where(result_df[:,2] == 5)[0])]
    # mask = np.ones(len(result_df), dtype=bool)
    # mask[np.where(result_df[:,2] == 5)[0]] = False
    # result_df = result_df[mask,...]

    #plot delta e hist for training validation and test set
    plot_hist_pred_delta_e(np.concatenate(results_train['DELTA']), 'Train', final_output_dir, 100)
    plot_hist_pred_delta_e(np.concatenate(results_valid['DELTA']), 'Validation', final_output_dir, 100)
    plot_hist_pred_delta_e(np.concatenate(results_test['DELTA']), 'Test', final_output_dir, 100)

    #plot boxplot
    plot_boxplot_pred_delta_e([
        np.concatenate(results_train['DELTA']),
        np.concatenate(results_valid['DELTA']),
        np.concatenate(results_test['DELTA'])],
      final_output_dir)

    #print mean, std
    print('Training delta E mean:', results_train['DELTA'].mean(), 'std:', results_train['DELTA'].std())
    print('Validation delta E mean:', results_valid['DELTA'].mean(), 'std:', results_valid['DELTA'].std())
    print('Test delta E mean:', results_test['DELTA'].mean(), 'std:', results_test['DELTA'].std())

    print('See results on {}'.format(final_output_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--illuminant', type=str, default='D65', help='illuminant to use [D65, F11, F4, ...]')
    parser.add_argument('--data_dir', type=str, default='multiviews', help='path to the dataset root')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory pathname')
    parser.add_argument('--n_views', type=int, default=16, choices=[16,8,4,3,2,1], help='number of views to use, 1 for single view model')
    parser.add_argument('--weights', type=str, help='path to the model weights', required=True)
    parser.add_argument('--mode', type=str, default='RGBD', help='RGBD or RGB mode', choices=['RGBD', 'RGB'])
    parser.add_argument('--test_on', type=str, choices=['subcolor', 'color', 'shape', 'folder_split'], default='subcolor', help="Select test set based on different subcolor/color/shape")
    parser.add_argument('--value', help='Value for test selection, can be a color (str), subcolor ([0 - 9]), shape (str)', default=5)

    opt = parser.parse_args()

    evaluate(opt)