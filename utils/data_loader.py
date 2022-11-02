from select import select
from matplotlib import test
import numpy as np
import os
import tensorflow as tf
import random 

def load_and_preprocess_rgbd(data_dir, n_views, img_height=128, img_width=128):

    #we expect n_views * 2 imgs (RGB + D)
    files = os.listdir(data_dir)
    
    print('{} images found'.format(n_views))

    #we need to know wich ones are depth map
    #we expect imagename.png/.jpg for rgb images and Z_depthImagename.png for depth maps

    rgb_d_list = []

    for f in files:
        z_f = f

        #just for blender name to remove Imagebe before the imgs name
        if 'Image' in f:
            z_f = f[5:]

        if 'Z_depth' not in f:
            z_name = 'Z_depth' + z_f

            #load rgb images and aggregate
            rgb_image = tf.keras.preprocessing.image.load_img(
            path= data_dir + '/' + f, grayscale=False, color_mode='rgb', target_size=(img_height, img_width),
            interpolation='nearest'
            )
    
            #load depth map
            d_image = tf.keras.preprocessing.image.load_img(
            path=data_dir + '/' + z_name, color_mode='grayscale', target_size=(img_height, img_width),
            interpolation='nearest'
            )

            d = tf.keras.preprocessing.image.img_to_array(d_image)
            rgb = tf.keras.preprocessing.image.img_to_array(rgb_image)

            #NORMALIZATION
            d /=  (2**16 - 1) # we use 16 bit int as png file - this will return the real value as mt in range [0,1] 
            rgb /= 255        # normalize RGB channels 

            rgb_d = np.concatenate([rgb, d], axis=2, dtype=np.float16) #the array is RGBD now
            rgb_d_list.append(rgb_d)

    return rgb_d_list

def load_and_preprocess_rgb(data_dir, n_views, img_height=128, img_width=128):

    #we expect n_views imgs (RGB)
    files = os.listdir(data_dir)
    
    print('{} images found'.format(n_views))

    rgb_list = []

    for f in files:
        if 'depth-' not in f:
            #load rgb images and aggregate
            rgb_image = tf.keras.preprocessing.image.load_img(
            path= data_dir + '/' + f, grayscale=False, color_mode='rgb', target_size=(img_height, img_width),
            interpolation='nearest'
            )

            rgb = tf.keras.preprocessing.image.img_to_array(rgb_image)

            #NORMALIZATION
            rgb /= 255        # normalize RGB channels 

            rgb_list.append(rgb)

    return rgb_list

def get_random_k_view(k, lower=0, upper=16):
    '''
        return k random idxs within the givenrange
        args:
            - k: int number of views 
            - lower: lower range value (included)
            - upper: upper tange value (not included)
    '''
    return random.sample(range(lower,upper), k)


def get_data_illuminant_data(data_path, illuminant):
    '''
        return the dataset for an illuminant as numpy array
        args:
            - data_path: str root path to the dataset
            - illuminant: str name of the illuminant
    '''

    # data schema 
    #TODO check shape is misssing 
    #edit it isnt missing but not added in the dtype (get error.. to see)
    df = np.array([],dtype=[('shape', 'U4'), ('color', 'U10'),('subcolor', np.int8), ('light', 'U4'), ('data', np.float16, (16, 128, 128, 4)),('LAB', np.float16, (3,))])

    illuminant_path = os.path.join(data_path, illuminant)
    shapes_list = os.listdir(illuminant_path)

    for shape in shapes_list:
        shape_path = os.path.join(illuminant_path, shape)
        df_list = os.listdir(shape_path)
        for color_df_name in df_list:
            path = os.path.join(shape_path, color_df_name)
            data = np.load(path, allow_pickle=True)

            if df.shape[0] == 0:
                df = data
            else:
                df = np.append(df, data, axis = 0)
    return df

# def get_test_set(df):
#     '''
#         remove and return the test set from df (subcolors 5)
#         note that subcolors 5 have been created randomly

#         args:
#             - df numpy array
#         return df, df_test: np array
#     '''
#     # get test set
#     df_test = df[np.unique(np.where(df['subcolor'] == 5)[0])]

#     #remove test data from df
#     mask = np.ones(len(df), dtype=bool)
#     mask[np.where(df['subcolor'] == 5)[0]] = False
#     df = df[mask,...]

#     return df, df_test

def get_test_set(df, select_on, value):
    '''
        remove and return the test set from df (subcolors 5)
        note that subcolors 5 have been created randomly

        args:
            - df numpy array
            - select_on str of the feature to use to split
            - value str or int: feature value to put in the test 
        return df, df_test: np array
    '''
    # get test set
    test_idxs = np.where(df[select_on] == value)[0]
    df_test = df[np.unique(test_idxs)]

    #remove test data from df
    mask = np.ones(len(df), dtype=bool)
    mask[test_idxs] = False
    df = df[mask,...]

    return df, df_test

def split_on_value(df, select_on, value, axis = None):
    '''
        Take ds on selected value given axis

        args:
            - df numpy array
            - select_on str of the feature to use to split
            - value str or int: feature value to put in the test 
            - axis
        return df, df_test: np array
    '''
    # get test set
    test_idxs = np.where(np.take(df, select_on, axis) == value)[0]
    df_test = df[np.unique(test_idxs)]

    #remove test data from df
    mask = np.ones(len(df), dtype=bool)
    mask[test_idxs] = False
    df = df[mask,...]

    return df, df_test

def get_labels(df):
    return np.array([ df['LAB'][i][0] for i in range(len(df))])

def split_dataset(df, output_dir, mode, select_test_on, validation_size=0.3):
    '''
        Split dataset into training, validation and test
        we use all subcolor 5 as test data.
        returns: 3 tuples  (X_train, y_train), (X_val, y_val), (X_test, Y_test)
        args:
            - df: numpy array with dtype dtype=[('color', 'U10'),('subcolor', np.int8), ('light', 'U4'), ('data', np.float16, (16, 128, 128, 4)),('LAB', np.float16, (3,))] 
            (i.e. obtained by get_data_illuminant_data)
            - output_dir: str output path 
            - validation_size: float number indicates the size of the validation set
            - random_state: int random state for train_test_split function
    '''

    select_on, val = select_test_on

    #get test set and remove test samples from df
    df, df_test = get_test_set(df, select_on, val)

    #get labels LAB values
    df_labels = get_labels(df)
    test_labels = get_labels(df_test)

    #NOTE it is after drop subcolors 5!
    shuffled_indexs = np.arange(0,len(df))
    np.random.shuffle(shuffled_indexs)

    offset = int(len(df)*validation_size)

    train_indexs = shuffled_indexs[offset:]
    validation_indexs = shuffled_indexs[:offset]

    X_train = df['data'][train_indexs]
    y_train = df_labels[train_indexs]

    X_val = df['data'][validation_indexs]
    y_val = df_labels[validation_indexs]

    X_test = df_test['data']

    if mode == 'RGB':
        X_train = X_train[:,:,:,:,:3]
        X_val = X_val[:,:,:,:,:3]
        X_test = X_test[:,:,:,:,:3]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'train_indexs'), train_indexs)
    np.save(os.path.join(output_dir, 'validation_indexs'), validation_indexs)

    return (X_train, y_train), (X_val, y_val), (X_test, test_labels)

def get_tensorflow_dataset(df, batch_size=32):
    '''
        return tf.data.Dataset
        args:
            - df: tuple with (X, Y)
            - batch_size: int
    '''
    ds = tf.data.Dataset.from_tensor_slices(df).shuffle(buffer_size=len(df[0])).batch(batch_size)

    return ds

def save_tf_dataset(dataset, path):
    dataset.save(path)

def load_tf_dataset(path):
    new_dataset = tf.data.Dataset.load(path)

    return new_dataset

def get_data_from_dir(data_path):
    '''
        return the dataset for an illuminant as numpy array
        args:
            - data_path: str root path to the dataset
            - illuminant: str name of the illuminant
    '''

    df = np.array([],dtype=[('shape', 'U4'), ('color', 'U10'),('subcolor', np.int8), ('light', 'U4'), ('data', np.float16, (16, 128, 128, 4)),('LAB', np.float16, (3,))])

    shapes_list = os.listdir(data_path)

    for shape in shapes_list:
        shape_path = os.path.join(data_path, shape)
        df_list = os.listdir(shape_path)
        for color_df_name in df_list:
            path = os.path.join(shape_path, color_df_name)
            data = np.load(path, allow_pickle=True)

            if df.shape[0] == 0:
                df = data
            else:
                df = np.append(df, data, axis = 0)
    return df

def get_X_and_Y(df, mode):
    X = df['data']
    Y = get_labels(df)

    if mode == 'RGB':
        X = X[:,:,:,:,:3]

    return (X,Y)

def split_dataset_from_directory(data_dir, mode='RGBD'):
    """
        return train test and split dataset
        
        args:
            - data_dir: str, path to root folder
             inside root folder it expects: 
             train valid and test folders that follow the usual scheme
    """
    train_dataset = get_data_from_dir(os.path.join(data_dir, 'train'))
    train_dataset = get_X_and_Y(train_dataset, mode)

    valid_dataset = get_data_from_dir(os.path.join(data_dir, 'valid'))
    valid_dataset = get_X_and_Y(valid_dataset, mode)

    test_dataset = get_data_from_dir(os.path.join(data_dir, 'test'))
    test_dataset = get_X_and_Y(test_dataset, mode)

    return train_dataset, valid_dataset, test_dataset

def get_dataset_end2end(data_path, illuminant, mode, batch_size, output_dir, select_test_on):
    if select_test_on[0] == 'folder_split':
        (X_train, y_train), (X_val, y_val), (X_test, Y_test) = split_dataset_from_directory(data_path, mode)
    else:
        df = get_data_illuminant_data(data_path, illuminant)
        (X_train, y_train), (X_val, y_val), (X_test, Y_test) = split_dataset(df, output_dir, mode, select_test_on, validation_size=0.3)
    
    del df

    train_dataset = get_tensorflow_dataset((X_train, y_train), batch_size=batch_size)
    validation_dataset = get_tensorflow_dataset((X_val, y_val), batch_size=batch_size)
    test_dataset = get_tensorflow_dataset((X_test, Y_test), batch_size=batch_size)

    del X_train, y_train, X_val, y_val, X_test, Y_test 

    return train_dataset, validation_dataset, test_dataset
