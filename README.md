# Bee Colony Optimizer of deep neural networks architectures for image classification



## Dependencies
To run this code, you will need the following packages installed on you machine:

- Python 3.7;
- Tensorflow 1.14;
- Keras 2.2.4;
- Numpy 1.16.4;
- Matplotplib 3.1.0.

**Note1:** If your system has all these packages installed, the code presented here should be able to run on Windows, macOS, or Linux.

## Usage

1. First, clone this repository:

	```
	git clone https://github.com/mohit6199/abc-CNN.git
	```

2. Download the following datasets and extract them to their corresponding folders inside the ```datasets``` folder:
	1. Convex: 
[http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip)
	2. Rectangles: [http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip)
	3. Rectangles with Background Images: [http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip)
	4. MNIST with Background Images: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip)
	5. MNIST with Random Noise as Background: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip)
	6. MNIST with Rotated Digits: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip)
	7. MNIST with Rotated Digits and Background Images: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip)


3. Now, you can test the algorithm by running the ```main.py``` file:

	```
	python main.py
	```
	
	or
	
	```
	python3 main.py
	```

**Note2:** The algorithm's parameters can modified in the file ```main.py```.

