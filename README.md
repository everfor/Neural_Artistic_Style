# Neural_Artistic_Style
- Implementation of paper [A Neural Algorithm of Artistic Style (arXiv:1508.06576)][paper] for re-construct an image with the art style of another image (Inspiration of codes drawn from [anishatahlye's neural-style][repo]. For a better (also the first) implementation in Tensorflow, please refer to his repository)
- Implementation of paper [Preserving Color in Neural Artistic Style Transfer (arXiv:1606.05897)][paper2] for the option of preserving color scheme of the original image when tranferring styles

# Dependecies
- Python 3+ (Might work with Python 2.7+, only tested in Python 3.5)
- Tensorflow
- Numpy
- Scipy
- <b>A pretrained VGG19 convnet</b>. The program uses [imagenet pretrained vgg19][vgg]. By default, the program assume the convnet lies within the same folder as paint_style.py

# Example Usage
`python3 paint_style.py --content CONTENT_IMAGE --style STYLE_IMAGE --output OUTPUT_IMAGE`

or

`python3 paint_style.py -c CONTENT_IMAGE -s STYLE_IMAGE -o OUTPUT_IMAGE`

Run with `--preserve-colors True` to preserve the original color schemes

More options exists. Run `python3 paint_style.py --help` for a decriptions of all available options.


# Demo
The program was tested converting the following image into different artistic styles:
![content](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/content.jpg)

- <b>The Starry Night, Vincent Van Gogh, 1989, Expressionism</b>

![out-1](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-1.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/the_starry_night.jpg" height="235"/>


- <b>Impression, Sunrise, Claude Monet, 1872, Impressionism</b>

![out-3](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-3.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/sunrise.jpg" height="235"/>


- <b>The Shipwreck of the Minotaur, Joseph Mallord William Turner, 1805, Romantism</b>

![out-2](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-2.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/the_shipwreck_of_the_minotaur.jpg" height="235"/>


- <b>Bottle and Fishes, Georges Braque, 1912, Cubism</b>

![out-4](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-4.jpg)
<img src="https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/bottle_and_fishes.jpg" height="235"/>


- <b>Color Scheme Preserving</b>

Preserve the original colr scheme while using the art style of "The Shipwreck of the Minotaur"

![content](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/content.jpg)
![out-5](https://github.com/everfor/Neural_Artistic_Style/blob/master/demo/out-5.jpg)

[paper]: https://arxiv.org/abs/1508.06576
[paper2]: https://arxiv.org/abs/1606.05897
[repo]: https://github.com/anishathalye/neural-style
[vgg]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
