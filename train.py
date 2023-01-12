import argparse
import tensorflow as tf
import pandas as pd
import os
import time
import yaml

from utils.data_loader import get_dataset_end2end, get_random_k_view
from models.model import build_model
from utils.metrics import delta_e, delta_e_2000
from utils.plot_utils import plot_loss, plot_de
# from utils.models_utils import predict_on_n_views

def checkpoint(chkpoint_manager, final_output_dir, history):
  print('Saving weights')
  os.makedirs(os.path.dirname(final_output_dir), exist_ok=True)
  history.to_csv(os.path.join(final_output_dir,'history.csv'), index=False)
  chkpoint_manager.save()

def save_new_best(model, final_output_dir):
  print('Saving new best model')
  os.makedirs(final_output_dir, exist_ok=True)
  model.save_weights(os.path.join(final_output_dir, 'model-best.h5'))

def print_gpu_memory_usage(msg):
    print(msg)
    if tf.config.list_physical_devices('GPU'):
        print('current', tf.config.experimental.get_memory_info('GPU:0')['current']/1024/1024/1024)
        print('peak', tf.config.experimental.get_memory_info('GPU:0')['peak']/1024/1024/1024)

def run(opt):

    illuminant = opt.illuminant
    data_dir = opt.data_dir
    batch = opt.batch
    output_dir = opt.output_dir
    n_views = opt.n_views
    epochs = opt.epochs
    lr = opt.lr
    mode = opt.mode

    final_output_dir = os.path.join(output_dir, illuminant, str(n_views) + '_views', mode, 'run')

    if os.path.exists(final_output_dir):
        run_list = os.listdir(os.path.join(output_dir, illuminant, str(n_views) + '_views', mode))
        i = len(run_list)
        final_output_dir = os.path.join(output_dir, illuminant, str(n_views) + '_views', mode, 'run' + str(i))
        print('Eperiment folder already exists - creating: {}'.format(final_output_dir))

    print('Experiments directory:', final_output_dir)
    os.makedirs(final_output_dir ,exist_ok=True)

    with open(os.path.join(final_output_dir, 'configuration.yaml'), 'w') as f:
      yaml.dump(
        {
          'illuminant':illuminant,
          'data_dir':data_dir,
          'batch':batch,
          'output_dir':final_output_dir,
          'n_views': n_views,
          'epochs': epochs,
          'lr': lr,
          'mode': mode,
          'test_on': opt.test_on,
          'value': opt.value
        }
        , f)

    select_test_on = (opt.test_on, opt.value)

    print_gpu_memory_usage("before data load")

    train_dataset, validation_dataset, test_dataset = get_dataset_end2end(data_dir, illuminant, mode, batch, final_output_dir, select_test_on)

    print_gpu_memory_usage("after data load")

    print('Building model...')
    model = build_model(n_views, mode)

    print(model.summary())

    print_gpu_memory_usage("after model load")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()

    loss_tracker = tf.keras.metrics.Mean(name='loss')
    val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')

    train_metric = tf.keras.metrics.Mean(name='DE')
    test_metric = tf.keras.metrics.Mean(name='test_DE')
    val_metric = tf.keras.metrics.Mean(name='val_DE')

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[loss_tracker, val_loss_tracker, train_metric, test_metric])

    history = pd.DataFrame(columns=['epoch', 'loss', 'validation_loss', 'DE', 'validation_DE', 'test_DE'])

    chkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    chkpoint_manager = tf.train.CheckpointManager(
        chkpoint, directory = os.path.join(final_output_dir, 'chkp/'), max_to_keep = 3, checkpoint_name='ckpt')

    stopped = 0 #start epoch

    # print('Initial LR',  optimizer._decayed_lr(var_dtype=tf.float32))

    #for ordered views
    views_dict_idxs = {16: range(16), 8:range(0,16,2), 4:[12, 14, 0, 2], 3:[12, 0, 4], 2:[0, 4]}

    minDE = float('inf')

    for epoch in range(stopped, epochs):

        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # TRAINING

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            inputs = [x_batch_train[:, i] for i in get_random_k_view(n_views)]

            with tf.GradientTape() as tape:
                if step == 0:
                    print_gpu_memory_usage("start step 0")

                #fixed positions
                # logits = model([x_batch_train[:, i] for i in views_dict_idxs[n_views]])

                # testing position indipendence and random views
                logits = model(inputs)

                loss_value = loss_fn(y_batch_train, logits)
                
                if step == 0:
                    print_gpu_memory_usage("end step 0")

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            loss_tracker.update_state(loss_value)
            train_metric.update_state(delta_e_2000(y_batch_train, logits))

        # Display metrics at the end of each epoch.
        train_loss = loss_tracker.result()
        train_de = train_metric.result()

        print("Training loss over epoch: {:.2f}, Training DE: {:.2f}".format( float(train_loss), float(train_de) ))

        # Reset training metrics at the end of each epoch
        loss_tracker.reset_states()
        train_metric.reset_states()

        # VALIDATION
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in validation_dataset:
            #fixed positions
            # val_logits = model([x_batch_val[:, i] for i in views_dict_idxs[n_views]], training=False)

            # testing position indipendence and random views
            val_logits = model([x_batch_val[:, i] for i in get_random_k_view(n_views)], training=False)

            # Update val metrics
            val_loss_tracker.update_state(loss_fn(y_batch_val, val_logits))
            val_metric.update_state(delta_e_2000(y_batch_val, val_logits))

        val_loss = val_loss_tracker.result()
        val_de = val_metric.result()

        val_loss_tracker.reset_states()
        val_metric.reset_states()

        print("Validation loss over epoch: {:.2f}, Validation DE: {:.2f}".format( float(val_loss), float(val_de)))

        # TEST
        #TODO make a function and call it here and outside the loop

        if opt.test == 2 or (opt.test == 1 and epoch + 1 == epochs):
            # Run a test loop depending on opt test (2 every epoch, 1 just at the last epoch )
            for x_batch_test, y_batch_test in test_dataset:
                # test_logits = predict_on_n_views(model, x_batch_test, n_views, training=False)
                # test_logits = model([x_batch_test[:, i] for i in range(16)], training=False)

                # fixed positions
                # test_logits = model([x_batch_test[:, i] for i in views_dict_idxs[n_views]], training=False)

                # testing position indipendence and random views
                test_logits = model([x_batch_test[:, i] for i in get_random_k_view(n_views)], training=False)


                # Update val metrics
                test_metric.update_state(delta_e_2000(y_batch_test, test_logits))

            test_de = test_metric.result()
            test_metric.reset_states()

            history = history.append({'epoch':epoch, 'loss': train_loss.numpy() , 'validation_loss':val_loss.numpy(), 'DE':train_de.numpy(), 'validation_DE':val_de.numpy(), 'test_DE': test_de.numpy()}, ignore_index=True)
            
            print("Test DE: {:.2f}".format( float(test_de)))

            if val_de.numpy() < minDE:
                minDE = val_de.numpy()
                save_new_best(model, final_output_dir)

            if epoch !=0 and epoch % 10 == 0:
                checkpoint(chkpoint_manager, final_output_dir, history)

        print("Time taken: %.2fs" % (time.time() - start_time))

    print('\nTraining finished')
    print('Min Delta E: {:.2f}'.format(minDE))
    print('Saving last model')
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_weights( os.path.join(final_output_dir,'model-last.h5'))
    history.to_csv(os.path.join(final_output_dir, 'history.csv'))

    plot_loss(history, final_output_dir)
    plot_de(history, final_output_dir)

    del model, train_dataset, test_dataset, validation_dataset
    return final_output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #TODO add others illuminants
    parser.add_argument('--illuminant', type=str, default='D65', help='illuminant to use [D65, F11, F4, ...], in case of folder_split it can be used to name the output dir')
    parser.add_argument('--data_dir', type=str, default='multiviews', help='path to the dataset root')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--output_dir', type=str, default='experiments', help='output directory pathname')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_views', type=int, default=16, help='number of views to use, 1 for single view model')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--test', type=int, choices=[0,1,2], default=2, help='0 dont test the model, 1 test at the end of training, 2 test after each epoch')
    parser.add_argument('--mode', type=str, default='RGBD', help='RGBD or RGB mode', choices=['RGBD', 'RGB'])
    parser.add_argument('--test_on', type=str, choices=['subcolor', 'color', 'shape', 'folder_split'], default='subcolor', help="Select test set based on different subcolor/color/shape")
    parser.add_argument('--value', help='Value for test selection, can be a color (str), subcolor ([0 - 9]), shape (str)', default=5)

    opt = parser.parse_args()

    run(opt)