import tensorflow as tf
import numpy as np
import scipy.misc
import os
import time
from glob import glob
from scipy import io

from tensorflow.python.platform import gfile
from tensorflow.python.keras import backend as K

def get_image(image_path, grayscale=True, train=True, noisy=True):
    image_list = []
    if train:
        if noisy:
            for path in image_path:
                image = io.loadmat(path)['img_noisy']
                image = np.reshape(image, [image.shape[0], image.shape[1], 1])
                image_list.append(image)
        else:
            for path in image_path:
                image = io.loadmat(path)['img_scale']
                image = np.reshape(image, [image.shape[0], image.shape[1], 1])
                image_list.append(image)
    else:
        image = io.loadmat(image_path)['Eabs']
        image = np.reshape(image, [-1, image.shape[0], image.shape[1], 1])
        image_list = image
    images = np.asarray(image_list)
    return images

def createimagelists(imageDir):
    pattern = os.path.join(imageDir, '*.' + 'mat')
    print(pattern)
    filelists = gfile.Glob(pattern)
    return filelists

def shufflefile(filelists, seed):
    np.random.seed(seed)
    np.random.shuffle(filelists)
    return filelists

def batch_gen(epochlist, clean_dir):
    clean_list = []
    for path in epochlist:
        filename = os.path.basename(path)
        clean_list.append(os.path.join(clean_dir, filename))
    return clean_list

def SNR_UNET(img):
    model_input = tf.keras.layers.Input(shape=(K.int_shape(img)[1], K.int_shape(img)[2], K.int_shape(img)[3],),
                                        tensor=img)

    conv1 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, padding='same', activation='relu', strides=1,
                                   kernel_initializer='glorot_uniform', name='conv1')(model_input)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv2')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    conv2_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv2a = tf.keras.layers.Conv2D(64, (7, 7), use_bias=False, padding='same', activation='relu', strides=1,
                                     kernel_initializer='glorot_uniform', name='conv2a')(model_input)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv3')(conv2_pool)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)

    conv4 = tf.keras.layers.Conv2D(128, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv4')(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    conv4_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv4a = tf.keras.layers.Conv2D(128, (7, 7), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv4a')(conv2_pool)
    conv4a = tf.keras.layers.BatchNormalization()(conv4a)
    conv4a = tf.keras.layers.Activation('relu')(conv4a)

    conv5 = tf.keras.layers.Conv2D(256, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv5')(conv4_pool)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation('relu')(conv5)

    conv6 = tf.keras.layers.Conv2D(256, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv6')(conv5)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation('relu')(conv6)
    conv6_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)

    conv6a = tf.keras.layers.Conv2D(256, (7, 7), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv6a')(conv4_pool)
    conv6a = tf.keras.layers.BatchNormalization()(conv6a)
    conv6a = tf.keras.layers.Activation('relu')(conv6a)

    conv7 = tf.keras.layers.Conv2D(512, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv7')(conv6_pool)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation('relu')(conv7)

    conv8 = tf.keras.layers.Conv2D(512, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv8')(conv7)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation('relu')(conv8)
    conv8_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv8)

    conv8a = tf.keras.layers.Conv2D(512, (7, 7), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv8a')(conv6_pool)
    conv8a = tf.keras.layers.BatchNormalization()(conv8a)
    conv8a = tf.keras.layers.Activation('relu')(conv8a)

    conv9 = tf.keras.layers.Conv2D(1024, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv9')(conv8_pool)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Activation('relu')(conv9)

    conv10 = tf.keras.layers.Conv2D(1024, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv10')(conv9)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    conv10 = tf.keras.layers.Activation('relu')(conv10)

    upconv1 = tf.keras.layers.Conv2DTranspose(512, (2, 2), use_bias=False, padding='same', activation='relu', strides=2,
                                             kernel_initializer='glorot_uniform', name='upconv1')(conv10)
    concat1 = tf.keras.layers.Concatenate()([upconv1, conv8, conv8a])
    conv11 = tf.keras.layers.Conv2D(512, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                   kernel_initializer='glorot_uniform', name='conv11')(concat1)
    conv11 = tf.keras.layers.BatchNormalization()(conv11)
    conv11 = tf.keras.layers.Activation('relu')(conv11)

    conv12 = tf.keras.layers.Conv2D(512, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv12')(conv11)
    conv12 = tf.keras.layers.BatchNormalization()(conv12)
    conv12 = tf.keras.layers.Activation('relu')(conv12)

    upconv2 = tf.keras.layers.Conv2DTranspose(256, (2, 2), use_bias=False, padding='same', activation='relu', strides=2,
                                              kernel_initializer='glorot_uniform', name='upconv2')(conv12)
    concat2 = tf.keras.layers.Concatenate()([upconv2, conv6, conv6a])
    conv13 = tf.keras.layers.Conv2D(256, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv13')(concat2)
    conv13 = tf.keras.layers.BatchNormalization()(conv13)
    conv13 = tf.keras.layers.Activation('relu')(conv13)

    conv14 = tf.keras.layers.Conv2D(256, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv14')(conv13)
    conv14 = tf.keras.layers.BatchNormalization()(conv14)
    conv14 = tf.keras.layers.Activation('relu')(conv14)

    upconv3 = tf.keras.layers.Conv2DTranspose(128, (2, 2), use_bias=False, padding='same', activation='relu', strides=2,
                                              kernel_initializer='glorot_uniform', name='upconv3')(conv14)
    concat3 = tf.keras.layers.Concatenate()([upconv3, conv4, conv4a])
    conv15 = tf.keras.layers.Conv2D(128, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv15')(concat3)
    conv15 = tf.keras.layers.BatchNormalization()(conv15)
    conv15 = tf.keras.layers.Activation('relu')(conv15)

    conv16 = tf.keras.layers.Conv2D(128, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv16')(conv15)
    conv16 = tf.keras.layers.BatchNormalization()(conv16)
    conv16 = tf.keras.layers.Activation('relu')(conv16)

    upconv4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), use_bias=False, padding='same', activation='relu', strides=2,
                                              kernel_initializer='glorot_uniform', name='upconv4')(conv16)
    concat4 = tf.keras.layers.Concatenate()([upconv4, conv2, conv2a])
    conv17 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv17')(concat4)
    conv17 = tf.keras.layers.BatchNormalization()(conv17)
    conv17 = tf.keras.layers.Activation('relu')(conv17)

    conv18 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, padding='same', activation=None, strides=1,
                                    kernel_initializer='glorot_uniform', name='conv18')(conv17)
    conv18 = tf.keras.layers.BatchNormalization()(conv18)
    conv18 = tf.keras.layers.Activation('relu')(conv18)

    conv19 = tf.keras.layers.Conv2D(1, (1, 1), use_bias=False, padding='same', activation='linear', strides=1,
                                    kernel_initializer='glorot_uniform', name='conv19')(conv18)

    conv20 = img - conv19

    return conv20

# optimizer
def optimizer(loss, var_list, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list, global_step=global_step)
    return optimizer

class SNR(object):
    def __init__(self, num_epochs, epoch_size, batch_size, log_every):
        self.num_epochs = num_epochs
        self.num_steps = num_epochs*epoch_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.log_every = log_every

        self._create_model()

    def _create_model(self, scope=None):
        self.i_sn = tf.placeholder(tf.float32, [None, None, None, 1], name="image")
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])

        with tf.variable_scope('SNR') as scope:
            self.G = SNR_UNET(self.i_sn)
            tf.add_to_collection("net", self.G)

        # Loss Functions
        self.l1 = tf.reduce_mean(tf.abs(self.G-self.x))
        self.x_gradient = tf.image.image_gradients(self.x)
        self.g_gradient = tf.image.image_gradients(self.G)
        self.gradient_loss = tf.reduce_mean(tf.abs(tf.subtract(self.g_gradient, self.x_gradient)))

        self.loss_g = self.l1*1.0 + self.gradient_loss*0.04

       # Trainable Variables
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SNR')

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.001, self.global_step, 6000, 0.8, staircase=True)

        # Optimizer
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate, self.global_step)

    def train(self):
        noisy_img_path = './scale/noisy'
        clean_img_path = './scale/clean'
        test_img_path = './Test_images'
        result_path = './Results_ray_sacle_mat'

        filelists = createimagelists(noisy_img_path)
        testlists = createimagelists(test_img_path)

        with tf.Session() as sess:
            self.patch_size = 96
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

            # TensorBoard
            tf.summary.scalar('L1', self.l1)
            # tf.summary.scalar('ms_ssim', self.ms_ssim)
            tf.summary.scalar('Loss', self.loss_g)
            tf.summary.scalar('Learning_rate', self.learning_rate)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join('./Tensorboard', result_path), sess.graph)

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            for epoch in range(self.num_epochs):
                epoch_file = shufflefile(filelists, epoch)
                for step in range(self.epoch_size):
                    start_time = time.time()
                    glo_step = epoch*self.epoch_size + step

                    # load image filename list randomly
                    inputfile = epoch_file[step*self.batch_size:(step+1)*self.batch_size]
                    noisy_img_list = inputfile
                    clean_img_list = batch_gen(inputfile, clean_img_path)

                    # load image files
                    i_sn = get_image(noisy_img_list)
                    x = get_image(clean_img_list, noisy=False)

                    # update network
                    K.set_learning_phase(True)
                    loss_g, _, summary = sess.run([self.loss_g, self.opt_g, merged], {self.i_sn: i_sn, self.x: x})
                    train_writer.add_summary(summary, glo_step)

                    if (step+1) % self.log_every == 0:
                        print('{}: G: {:0.6f}'.format(glo_step, loss_g))

                        if (step+1) == self.epoch_size:
                            # Sampling train Images
                            K.set_learning_phase(False)
                            for j in range(3):
                                image_origin = np.reshape(x[j], [self.patch_size, self.patch_size])
                                scipy.misc.toimage(image_origin).save(
                                    os.path.join(result_path, ('img{}-{}_original.png'.format(str(glo_step).zfill(6), str(j)))))
                                image_noisy = np.reshape(i_sn[j], [self.patch_size, self.patch_size])
                                scipy.misc.toimage(image_noisy).save(
                                    os.path.join(result_path, ('img{}-{}_noisy.png'.format(str(glo_step).zfill(6), str(j)))))
                                noise = image_noisy - image_origin + 128
                                scipy.misc.toimage(noise).save(
                                    os.path.join(result_path, ('img{}-{}_noise.png'.format(str(glo_step).zfill(6), str(j)))))
                                i_sn_ = np.reshape(i_sn[j],[-1,self.patch_size, self.patch_size,1])
                                image_denoised = sess.run(self.G, feed_dict={self.i_sn: i_sn_})
                                image_denoised = np.reshape(image_denoised, [self.patch_size, self.patch_size])
                                scipy.misc.toimage(image_denoised).save(
                                    os.path.join(result_path, ('img{}-{}_denoised.png'.format(str(glo_step).zfill(6), str(j)))))
                                scipy.misc.toimage(image_denoised - image_origin + 128).save(
                                    os.path.join(result_path, ('img{}-{}_error.png'.format(str(glo_step).zfill(6), str(j)))))


                            for j, testfile in enumerate(testlists):
                                data_test = get_image(testfile, train=False)
                                image = sess.run(self.G, feed_dict={self.i_sn: data_test})
                                io.savemat(os.path.join(result_path, 'test_result{}-{}.mat'.format(str(j), str(glo_step).zfill(6))),
                                           mdict={'img_denoised': image})

                            print('Loop Time', round(time.time()-start_time, 3))

                if (epoch+1) % 5 == 0:
                    saver.save(sess, os.path.join('./Unet_model', result_path), global_step=self.global_step)
            train_writer.close()

def main():
    # with tf.device('/cpu:0'):
    model = SNR(
        35,  # training epochs
        6000, # epoch size
        64,  # batch size per data set
        100,  # log step
    )
    model.train()

if __name__ == '__main__':
    main()
