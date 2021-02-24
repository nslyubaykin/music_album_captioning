# Music Album Style Image Quoting with Transformers

This reporitory contains a TensorFlow [implementation](https://github.com/nslyubaykin/music_album_captioning/blob/master/albums_quoting.ipynb) of Transformers for music albums captioning. This model is more of a demonstaration so it has few parameters and is trained for only 10 epochs. 

# Data:

Model is trained on 330k music albums images and their captions (artist's & album's names concatenated with adding SEP token) parsed from the web. In the future, after tidying the data it will be published. Captions were tokenized with [Youtokentome](https://github.com/VKCOM/YouTokenToMe) BPE tokenizer.

# Model:

As a feature extractor and encoder ImageNet pretrained MobileNetV2 is used. Encoder maps raw image into a sequence of 49 tokens (each token is encoded with 128-dimensional vector). Last layer adds positional encoding for processing in the transformer decoder. As a decoder part classic Transformer decoder with self-attention is used. Trained models are attached and may be loaded from the checkpoints file.

# Captioning OOS Music Albums Covers:

![Album 1](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/alb1.png)

![Album 2](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/alb2.png)

![Album 3](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/alb3.png)

![Album 4](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/alb4.png)

# Captioning Arbitary User Images:

You can upload your own images to user_imgs folder and process them.

![User Image 1](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/uimg1.png)

![User Image 2](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/uimg2.png)

![User Image 3](https://github.com/nslyubaykin/music_album_captioning/blob/master/sample_imgs/uimg3.png)
