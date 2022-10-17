import matplotlib.pyplot as plt
import numpy as np
import os

def check_dir(output_dir):
    os.makedirs(os.path.join(output_dir,'imgs'), exist_ok=True)


def plot_sample(sample, output_dir, N=16):
    '''
        plot a sample RGB and D channels in 2 rows of N images
        args:
            -sample: numpy array with RGBD channel
            -N: int number of views
    '''
    fig, axs = plt.subplots(2, N, figsize=(36, 12), constrained_layout=True)

    for i, ax in enumerate(axs.flat):
        if i<N:
            img = sample[i % N,:,:,:3]
            ax.imshow(np.array(img, dtype=np.float32))
        else: 
            z = sample[i % N,:,:,3]
            ax.imshow(np.array(z,dtype=np.float32))
    check_dir(output_dir)
    plt.savefig(os.path.join(output_dir, 'imgs' ,'sample.png'))

def plot_loss(history, output_dir):
    plt.figure(figsize=(16,12))
    plt.plot(history['epoch'], history['validation_loss'], label='validation_loss', color='orange')
    plt.plot(history['epoch'], history['loss'], label='train_loss', color='blue')
    plt.legend()
    plt.title('Loss')

    check_dir(output_dir)
    plt.savefig(os.path.join(output_dir, 'imgs', 'loss.png'))

def plot_de(history, output_dir):
    plt.figure(figsize=(16,12))
    plt.plot(history['epoch'], history['validation_DE'], label='valid $\Delta$E', color='orange')
    plt.plot(history['epoch'], history['test_DE'], label='test $\Delta$E', color='red')
    plt.plot(history['epoch'], history['DE'], label='train $\Delta$E2', color='blue')
    plt.legend()
    plt.title('CIEDE2000')
    check_dir(output_dir)
    plt.savefig(os.path.join(output_dir,'imgs' ,'de.png'))

def plot_hist_pred_delta_e(arr, split_name, output_dir, n_bins=100):
    plt.figure(figsize=(16,12))
    plt.hist(arr,bins=n_bins)
    plt.title(split_name + ' $\Delta$E2000')
    check_dir(output_dir)
    plt.savefig(os.path.join(output_dir, 'imgs' ,'{}-hist.png'.format(split_name)))

def plot_boxplot_pred_delta_e(arr, output_dir):
    plt.figure(figsize=(16,12))
    plt.boxplot(arr, labels=['Train', 'Valid', 'Test'])
    plt.title('$\Delta$E2000')
    check_dir(output_dir)
    plt.savefig(os.path.join(output_dir,'imgs','box-plot.png'))

def plot_hist(arr, bins, title, output_dir):
    plt.figure(figsize=(16,12))
    plt.hist(arr, bins=bins)
    plt.title(title)
    check_dir(output_dir)
    plt.savefig( os.path.join(output_dir,'imgs','{}-hist.png'.format(title)))