- [COLOR PROJ repo](#color-proj-repo)
- [REPO FILES](#repo-files)
- [USAGE](#usage)
  - [TRAIN ON THE MULTIVIEW COLOR DATASET](#train-on-the-multiview-color-dataset)
    - [ARGUMENTS](#arguments)
  - [EVALUATE THE MODEL](#evaluate-the-model)
    - [ARGUMENTS](#arguments-1)
    - [COMPARE WITH OTHER METHODS](#compare-with-other-methods)
      - [ARGUMENTS](#arguments-2)
  - [PREDICT ON NEW IMAGES](#predict-on-new-images)
    - [PREPARE YOUR IMAGES](#prepare-your-images)
    - [RUN PREDICT](#run-predict)
    - [ARGUMENTS](#arguments-3)
  - [TUTORIAL](#tutorial)

# COLOR PROJ repo
This is the **CIELab color measurement through RGB-D images** official repo. (refs to paper to be added)

# REPO FILES
<pre>
&#128193; <a href=#>Color Proj</a>
   &#x1F5CE; <a href=./evaluate.py>evaluate.py</a>
   &#x1F5CE; <a href=./README.md>README.md</a>
   &#x1F5CE; <a href=./predict.py>predict.py</a>
   &#x1F5CE; <a href=./requirements.txt>requirements.txt</a>
   &#x1F5CE; <a href=./train.py>train.py</a>
  &#128193; <a href=./utils>utils</a>
      &#x1F5CE; <a href=./utils/metrics.py>metrics.py</a>
      &#x1F5CE; <a href=./utils/generate_dataset.py>generate_dataset.py</a>
      &#x1F5CE; <a href=./utils/data_loader.py>data_loader.py</a>
      &#x1F5CE; <a href=./utils/models_utils.py>models_utils.py</a>
      &#x1F5CE; <a href=./utils/plot_utils.py>plot_utils.py</a>
      &#x1F5CE; <a href=./utils/color_transform_utils.py>color_transform_utils.py</a>
  &#128193; <a href=./models>models</a>
      &#x1F5CE; <a href=./models/model.py>model.py</a>
</pre>


# USAGE

## TRAIN ON THE MULTIVIEW COLOR DATASET
Since the dataset cannot be downloaded automatically, please download it handly (refs to dataset to be added).

In order to train the model run train.py.

To reproduce the paper's results use the following arguments

```
python train.py --illuminant F11 --data_dir path/to/multiviews --epochs 50 --n_views 16
```

### ARGUMENTS


The possible arguments are:


```
--illuminant: illuminant to use, default='D65', choices=['D50', 'D65', 'F1', 'F11', 'F4', 'FF', 'LED-B1', 'LED-V1']
--data_dir: path to the dataset, default='multiviews', 
--batch, batch: size, default=32
--output_dir: output directory path, default='experiments'
--epochs: number of epochs  default=100
--n_views: number of views to use, to know more look at the paper, default=16, choices=[16,8,4,3,2,1]
--lr: initial learning rate, default=1e-3
--mode: RGBD or RGB mode, default=RGBD
--test_on: Select test set based on different subcolor/color/shape, type=str, choices=['subcolor', 'color', 'shape'], default='subcolor'
--value: Value for test selection, can be a color (str), subcolor ([0 - 9]), shape (str), default=5
```

## EVALUATE THE MODEL

To evaluate the model run evaluate.py

It will generate different images such as boxplot, hist and a pickle file containing the prediction and the corrisponding Delta E for each color.

```
python evaluate.py --illuminant F11 --data_dir path/to/multiviews --n_views 16 --weights path/to/model-best.h5
```

### ARGUMENTS

The possible arguments are:

```
--illuminant: illuminant to use, default='D65', choices=['D50', 'D65', 'F1', 'F11', 'F4', 'FF', 'LED-B1', 'LED-V1']
--data_dir: path to the dataset, default='multiviews', 
--output_dir: output directory path, default='experiments'
--n_views: number of views to use, to know more look at the paper, default=16, choices=[16,8,4,3,2,1]
--weights': path to the model's weights, required
--mode: RGBD or RGB mode, default=RGBD
--test_on: Select test set based on different subcolor/color/shape, type=str, choices=['subcolor', 'color', 'shape'], default='subcolor'
--value: Value for test selection, can be a color (str), subcolor ([0 - 9]), shape (str), default=5
```

### COMPARE WITH OTHER METHODS
To get the predctions with the formulation run **color_transform_utils.py**. 

It will generate plots and formula_predictions.pkl, note that this formula supports only D65 and D50.

Others methods will be added soon.


```
python utils/color_transform_utils.py --output_dir /experiments/F11/16_views/run --predictions_path experiments/F11/16_views/run/predictions.pkl
```

#### ARGUMENTS
```
--illuminant, type=str, default='D65', illuminant to use in the formula, choices=['D65', 'D50']
--data_dir, type=str, default='multiviews', path to the dataset root
--output_dir, type=str, default='experiments', output directory pathname
--predictions_path, type=str, path to model predictions pkl, required
```


## PREDICT ON NEW IMAGES

To predict the CIELab value on your rgb-d images use predict.py

### PREPARE YOUR IMAGES

The code except to find in the data-dir folder **n_view * 2 images**: n_views rgb images and n_views depth maps. 

Remember to use **PNG** images using **16 bit** for depth maps images (this is important due the normalization process).

Please rename the depth maps images by adding "depth-" as prefix of the corrispondent rgb image name.

i.e. for *image001.png* the depth map image must be named *depth-image0001.png*


### RUN PREDICT 

Example:

```
python predict.py --weights /content/experiments/F11/16_views/run/model-best.h5 --n_views 16 --data_dir /content/drive/MyDrive/clear/color_2
```

### ARGUMENTS
The list of the possible arguments is the following:

    --data_dir, type=str, path to the data, required
    --output_dir, type=str, default='experiments', output directory pathname
    --n_views, type=int, default=16, choices=[16,8,4,3,2,1], number of views to use
    --weights, type=str, path to the model weights, required

## TUTORIAL 

A tutorial is available in the following google colab notebook (link to be added).