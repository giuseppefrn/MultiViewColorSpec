import os
import argparse
import yaml
import shutil
import numpy as np
import pandas as pd
from skimage import color

from utils.data_loader import get_data_illuminant_data
from utils.plot_utils import plot_hist

from train import run
from evaluate import evaluate

def plot_model_delta_hist(opt):

    for split in ['train', 'valid', 'test']:
        #load pkl with 
        colorNet_preds_ds = pd.read_pickle(os.path.join(opt.predictions_path,'predictions-' + split + '.pkl'))

        #plot and save hist
        colorNet_delta = np.concatenate(colorNet_preds_ds['DELTA'])

        plot_hist(colorNet_delta, 50, '{}-Model-Delta'.format(split),opt.output_dir)

def predict_by_formula(opt):
    
    #dictonary to select a ROI for each shape
    mask_dict = {'Toru': {'x':(73,78), 'y':(65,70), 'view': 2}, 'Cube': {'x':(60,65), 'y':(65,70), 'view': 15}, 'Suza': {'x':(60,65), 'y':(60,65), 'view': 12},
                'Cone': {'x':(65,70), 'y':(70,75), 'view': 2}, 'Icos': {'x':(65,70), 'y':(60,65), 'view': 2}, 'Scul': {'x':(65,70), 'y':(60,65), 'view': 2},
                }

    df = get_data_illuminant_data(opt.data_dir, 'D65')

    res = pd.DataFrame(columns=['shape', 'color', 'subcolor','LAB', 'predicted LAB' ,'Delta E'])

    shapes = np.unique(df[:]['shape'])
    colors = np.unique(df[:]['color'])
    sub_colors = np.unique(df[:]['subcolor'])

    for s in shapes:
        s_ds = df[df[:]['shape'] == s]
        for c in colors:
            c_ds = s_ds[s_ds[:]['color'] == c]
            for sb in sub_colors:
                sb_ds = c_ds[c_ds[:]['subcolor'] == sb]

                # get ROI
                mask = np.zeros(shape=(128,128,3))
                x_lower = mask_dict[s]['x'][0]
                x_upper = mask_dict[s]['x'][1]

                y_lower = mask_dict[s]['y'][0]
                y_upper = mask_dict[s]['y'][1]

                assert sb_ds['data'].shape[0] > 0, 'Erorr empty sb_ds: shape {}, color {}, sub: {}, '.format(s, c, sb)
                
                view = mask_dict[s]['view']

                img = sb_ds['data'][view,:,:,:3]
                # print(img.shape)

                roi = img[y_lower:y_upper, x_lower:x_upper]
                # print(roi.shape)

                pred_lab = color.rgb2lab(roi, illuminant=opt.illuminant)
                pred_lab = np.mean(pred_lab, axis=(0,1))

                actual_lab = sb_ds['LAB'][0]

                delta_e2000 = color.deltaE_ciede2000(actual_lab, pred_lab)

                res = res.append({'shape': s, 'color':c, 'subcolor':sb, 'LAB': actual_lab, 'predicted LAB':pred_lab, 'Delta E': delta_e2000},ignore_index=True)

    os.makedirs(opt.output_dir, exist_ok=True)
    res.to_pickle(os.path.join(opt.output_dir , 'formula_predictions.pkl'))

    plot_hist(res['Delta E'], 50, 'Formula-Delta',opt.output_dir)

    return

class DictObj:
  def __init__(self, in_dict:dict):
    assert isinstance(in_dict, dict)
    for key, val in in_dict.items():
      if isinstance(val, (list, tuple)):
        setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
      else:
        setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #TODO add others illuminants
    parser.add_argument('--base_dir', type=str, required=True, help='base dir')
    opt = parser.parse_args()

    base_dir = opt.base_dir
    conf_dir = os.path.join(base_dir, 'confs')
    confs = os.listdir(conf_dir)

    for conf in confs:
      if conf[0] != '.':
        with open(os.path.join(conf_dir, conf), 'r') as f:
            c = yaml.safe_load(f)

            c1 = DictObj(c)
            output_dir = run(c1)

            c['output_dir'] = output_dir
            c['weights'] = os.path.join(output_dir, 'model-best.h5')
            
            c1 = DictObj(c)
            evaluate(c1)

            c['predictions_path'] = output_dir
            
            c1 = DictObj(c)
            
            if c1.illuminant == 'D65':
              predict_by_formula(c1)

            plot_model_delta_hist(c1)
            
            if os.path.exists(os.path.join(output_dir, 'chkp')):
              shutil.rmtree(os.path.join(output_dir, 'chkp'))
        os.remove(os.path.join(conf_dir, conf))
