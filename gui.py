from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import queue
import utils

UPDATE_OUTPUT_IMG_MSG = 'display_out'

IMAGE_DISPLAY_SIZE = 256


class PaintGUI(object):

    def __init__(self):
        self.root = Tk()
        self.root.geometry('600x700')
        self.root.title('Image Style Transfer')
        self.q = queue.Queue()
        self.output_path = None
        self.progressbar_value = 0

        # frame1 for displaying content_img and style_img select button
        self.fm1 = Frame(self.root, width=600, height=20)
        # content img select button
        self.content_button = Button(self.fm1, text='Content Img', command=self.load_content)
        self.content_hint_str = StringVar()
        self.content_button.pack(side=LEFT)
        # style img select button
        self.style_button = Button(self.fm1, text='Style Img', command=self.load_style)
        self.style_button.pack(side=RIGHT)
        self.fm1.pack(side=TOP, ipadx=100)

        # frame2 for displaying content image and style image
        self.fm2 = Frame(self.root, width=600, height=260)
        self.content_label = None
        self.style_label = None
        self.fm2.pack()

        # frame3 for displaying progress bar
        self.fm3 = Frame(self.root, width=600, height=50)
        self.progressbar = None
        self.progress_label = None
        self.progress_str = StringVar(value='Transferring 0%')
        self.fm3.pack(pady=10)

        # frame4 for displaying output image
        self.fm4 = Frame(self.root, width=600, height=260)
        self.output_label = None
        self.fm4.pack(pady=10)

        # frame5 for transfer and cancel button
        self.fm5 = Frame(self.root, width=600, height=20)
        # start transfer button
        self.transfer_button = Button(self.fm5, text='Transfer', command=self.start_transfer)
        self.transfer_button.config(state=DISABLED)
        self.transfer_button.pack(side=LEFT)
        # cancel button
        self.cancel_button = Button(self.fm5, text='Cancel', command=self.cancel)
        self.cancel_button.pack(side=RIGHT)
        self.fm5.pack(side=BOTTOM, ipadx=150)

    def update_img(self, type, img_path):
        img = Image.open(img_path)
        img = img.resize((IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE))
        tk_image = ImageTk.PhotoImage(img)
        if type is 'content':
            if self.content_label is None:
                self.content_label = Label(self.fm2, text='Content Image',
                                           textvariable=self.content_hint_str)
                self.content_label.pack(side=LEFT)
            is_valid = utils.check_content_img(img_path)
            if is_valid:
                if self.content_label is None:
                    self.content_label = Label(self.fm2, image=tk_image)
                    self.content_label.pack(side=LEFT)
                else:
                    self.content_label.config(image=tk_image)
                self.transfer_button.config(state=NORMAL)
            else:
                if self.content_label is not None:
                    self.content_label.pack_forget()
                text = 'The content image\'s width and height must be smaller than 600'
                self.content_label = Label(self.fm2, text=text)
                self.content_label.pack(side=LEFT)
                self.transfer_button.config(state=DISABLED)
        elif type is 'style':
            if self.style_label is None:
                self.style_label = Label(self.fm2, image=tk_image)
                self.style_label.pack(side=RIGHT)
            else:
                self.style_label.config(image=tk_image)
        elif type is 'output':
            if self.output_label is None:
                self.output_label = Label(self.fm4, text='Output Image', image=tk_image)
            else:
                self.output_label.config(image=tk_image)
            self.output_label.pack(side=TOP)

        self.root.mainloop()

    def register_callback(self, callback):
        self.callback = callback

    def start(self):
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None

    def load_content(self):
        self.callback('content')
        
    def load_style(self):
        self.callback('style')
        
    def start_transfer(self):
        self.callback('transfer')

    def cancel(self):
        self.callback('cancel')

    def process_incoming(self):
        while self.q.qsize():
            try:
                msg = self.q.get(0)
                print('incoming msg: ', msg)
                if msg is UPDATE_OUTPUT_IMG_MSG:
                    self.update_img('output', self.output_path)
            except queue.Empty:
                pass

    def reset_output_img(self):
        if self.output_label is not None:
            self.output_label.pack_forget()

    def update_buttons(self, is_enable):
        if is_enable:
            self.content_button.config(state=NORMAL)
            self.style_button.config(state=NORMAL)
            self.transfer_button.config(state=NORMAL)
        else:
            self.content_button.config(state=DISABLED)
            self.style_button.config(state=DISABLED)
            self.transfer_button.config(state=DISABLED)

    def push_msg(self, msg, arg):
        if msg is UPDATE_OUTPUT_IMG_MSG:
            self.output_path = arg
        self.q.put(msg)

    def period_call(self):
        self.root.after(200, self.period_call)
        self.process_incoming()

    def update_progressbar(self, value=0):
        if self.progressbar is None:
            # initialize progressbar
            self.progressbar = ttk.Progressbar(self.fm3, length=500, mode="determinate", orient=HORIZONTAL)
            self.progressbar.pack(side=TOP, anchor=CENTER)
            self.progressbar['maximum'] = 100
            self.progressbar['value'] = 0
            # initialize progress label
            self.progress_label = Label(self.fm3, textvariable=self.progress_str)
            self.progress_label.pack(side=TOP, anchor=W)
        else:
            self.progressbar['value'] = value
            if value != 100:
                self.progress_str.set('Transferring ' + str(value) + '%')
            else:
                self.progress_str.set('Transferred!')

    def start_gui(self, callback):
        self.register_callback(callback)
        self.period_call()
        self.start()
