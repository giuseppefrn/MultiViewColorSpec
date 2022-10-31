import os
import yaml
import shutil

from train import run
from evaluate import evaluate
from utils.color_transform_utils import predict_by_formula, plot_model_delta_hist

if __name__ == '__main__':
    base_dir = '/scratch/gfurnari'
    conf_dir = os.path.join(base_dir, 'confs')
    confs = os.listdir('/scratch/gfurnari/confs')

    for conf in confs[:1]:
        with open(os.path.join(conf_dir, conf), 'r') as f:
            c = yaml.safe_load(f)

            output_dir = run(c)

            c['output_dir'] = output_dir
            c['weights'] = os.path.join(output_dir, 'model-best.h5')

            evaluate(c)

            c['predictions_path'] = output_dir

            predict_by_formula(c)
            plot_model_delta_hist(c)

            shutil.rmtree(os.path.join(output_dir, 'chkp'))
