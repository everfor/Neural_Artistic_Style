# Neural_Artistic_Style
Implementation of paper [A Neural Algorithm of Artistic Style (arXiv:1508.06576)][paper] in Tensorflow.
Inspiration of codes drawn from [anishatahlye's neural-style][repo]. For a better (also the first) implementation in Tensorflow, please refer to his repository.

# Dependecies
- Python 3+ (Might work with Python 2.7+, only tested in Python 3.5)
- Tensorflow
- Numpy
- Scipy
- <b>A pretrained VGG19 convnet</b>. The program uses [imagenet pretrained vgg19][vgg]. By default, the program assume the convnet lies within the same folder as paint_style.py

# Example Usage
`python3 paint_style.py --content CONTENT_IMAGE --style STYLE_IMAGE --output OUTPUT_IMAGE`

or

`python3 paint_style.py -c CONTENT_IMAGE - STYLE_IMAGE -o OUTPUT_IMAGE`

More options exists. Run `python3 paint_style.py --help` for a decriptions of all available options.


# Demo
The program was tested converting the following image into different artistic styles:
![content](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/content.jpg)

- <b>The Starry Night, Vincent Van Gogh, 1989, Expressionism</b>

![out-1](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-1.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/the_starry_night.jpg" height="235"/>


- <b>Impression, Sunrise, Calude Monet, 1872, Impressionism</b>

![out-1](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-3.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/sunrise.jpg" height="235"/>


- <b>The Shipwreck of the Minotaur, Joseph Mallord William Turner, 1805, Romantism</b>

![out-1](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-2.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/the_shipwreck_of_the_minotaur.jpg" height="235"/>


- <b>Bottle and Fishes, Georges Braque, 1912, Cubism</b>

![out-1](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-4.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/bottle_and_fishes.jpg" height="235"/>


[paper]: https://arxiv.org/abs/1508.06576
[repo]: https://github.com/anishathalye/neural-style
[vgg]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
