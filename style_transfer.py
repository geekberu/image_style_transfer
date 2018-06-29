import os
import threading
import tensorflow as tf
import vgg
import cv2
import utils
import gui
from tkinter.filedialog import askopenfilename
from tkinter import *

slim = tf.contrib.slim

MAX_STEPS = 500
# MAX_STEPS = 20
VGG_MODEL_PATH = 'vgg_16.ckpt'

LEARNING_RATE = 1e1
W_STYLE = 500

paint = gui.PaintGUI()
content_input = None
style_input = None


def callback(type):
    if type == 'transfer':
        transfer_callback()
    elif type == 'content':
        global content_input
        # return file full path
        fname = askopenfilename(title='Select content image',
                                filetypes=[('image', '*.jpg'),
                                           ('All Files', '*')])
        if fname == '':
            return
        content_input = fname
        print('content image path: ', content_input)
        paint.update_img('content', content_input)

    elif type == 'style':
        global style_input
        # return file full path
        fname = askopenfilename(title='Select style image')
        if fname == '':
            return
        style_input = fname
        print('style image path: ', style_input)
        paint.update_img('style', style_input)

    elif type == 'cancel':
        sys.exit()


def style_transfer_gui():
    paint.start_gui(callback)


def transfer_callback():
    t = threading.Thread(target=start_transfer, name='TransferThread')
    t.setDaemon(True)
    t.start()


def start_transfer():
    if content_input is None:
        print('please input content image!!')
        return
    if style_input is None:
        print('please input style image!!')
        return
    print('start transferring...')
    paint.reset_output_img()
    paint.update_buttons(is_enable=False)
    paint.update_progressbar()
    with tf.Session() as sess:
        model = Train(sess)
        model.build_model()
        model.train()


class Train(object):
    def __init__(self, sess):
        self.sess = sess

    def build_model(self):
        # get content_img, style_img and define gen_img
        if content_input is not None:
            self.content_path = content_input
        if style_input is not None:
            self.style_path = style_input

        content_img, content_shape = utils.load_content_img(self.content_path)
        style_img = utils.load_style_img(self.style_path, content_shape)
        content_img_shape = content_img.shape
        gen_img = tf.Variable(tf.random_normal(content_img_shape) * 0.256)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([gen_img, content_img, style_img], axis=0))

            # calculate content_loss and style_loss
            content_loss = utils.cal_content_loss(f3)
            style_loss = utils.cal_style_loss(f1, f2, f3, f4)

            # load vgg model
            vgg_model_path = VGG_MODEL_PATH
            vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
            init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
            init_fn(self.sess)
            print('vgg_16 weights load done')

            self.gen_img = gen_img
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.content_loss = content_loss
            self.style_loss = style_loss * W_STYLE
            # the total loss
            self.loss = self.content_loss + self.style_loss

            # starter_learning_rate = 1e1
            # global_step = tf.train.get_global_step()
            # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps=500,
            #                                            decay_rate=0.98,
            #                                            staircase=True)
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss, global_step=self.global_step,
                                                                      var_list=gen_img)

        all_var = tf.global_variables()
        init_var = [v for v in all_var if 'vgg_16' not in v.name]
        init = tf.variables_initializer(var_list=init_var)
        self.sess.run(init)

        self.save = tf.train.Saver()

    def train(self):
        print('start training...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                # start_time = time.time()
                _, loss, step, cl, sl = self.sess.run([self.opt, self.loss, self.global_step,
                                                       self.content_loss, self.style_loss])

                if step % 100 == 0:
                # if step % 10 == 0:
                    if not os.path.exists('gen_img'):
                        os.mkdir('gen_img')
                    gen_img_list = tf.unstack(self.gen_img, axis=0)
                    gen_img = gen_img_list[0]
                    gen_img = self.sess.run(gen_img)
                    save_path = './gen_img/{}.jpg'.format(int(step/100))
                    # save_path = './gen_img/{}.jpg'.format(int(step/10))
                    cv2.imwrite(save_path, gen_img)
                    # update output image
                    paint.push_msg(gui.UPDATE_OUTPUT_IMG_MSG, save_path)

                # update progressbar
                value = step * 100 / MAX_STEPS
                paint.update_progressbar(int(value))

                print('[{}/{}],loss:{}, content:{},style:{}'.format(step, MAX_STEPS, loss, cl, sl))

                if step >= MAX_STEPS:
                    print('the training is finished!!')
                    paint.update_buttons(is_enable=True)
                    break

        except tf.errors.OutOfRangeError:
                self.save.save(self.sess, os.path.join(os.getcwd(), 'style-transfer-model.ckpt-done'))
                paint.update_buttons(is_enable=True)
        finally:
            coord.request_stop()
            paint.update_buttons(is_enable=True)
        coord.join(threads)


if __name__ == '__main__':
    style_transfer_gui()

