# A Neural Algorithm for Artistic Style Transfer with GUI
An implementation of Artistic Style Transfer algorithm using neural networks based on TensorFlow framework. I've also provided the GUI so it's easy to use.
## References
- **Paper**:
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576 "A Neural Algorithm of Artistic Style")

## Dependencies
- Python 3.6
- TensorFlow 1.12.0
- cv2
- tkinter
- PIL
- numpy
- [VGG16 pre-trained Model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz "VGG16 pre-trained Model")

## Training
`python style_transfer.py`

- It will popup the GUI, click the "Content Img" button and "Style Img" button to choose content and style, then click "Transfer", ok, it's training now. Very easy!

![](https://github.com/geekberu/image_style_transfer/blob/master/examples/gui/gui1.jpg?raw=true)

- After 500 steps iteration the training is finished just check the result on the GUI

![](https://github.com/geekberu/image_style_transfer/blob/master/examples/gui/gui2.jpg?raw=true)

## Test Results
<div align="center">
 <img src="https://github.com/geekberu/image_style_transfer/blob/master/examples/content_2.jpg?raw=true" height="223px">
 <img src="https://github.com/geekberu/image_style_transfer/blob/master/examples/style_4.jpg?raw=true" height="223px">
</div>
<div align="center">
<img src="https://github.com/geekberu/image_style_transfer/blob/master/examples/gen_img_2_4.jpg?raw=true" width="423px">
</div>