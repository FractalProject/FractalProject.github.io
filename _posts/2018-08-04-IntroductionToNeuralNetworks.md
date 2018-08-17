---
layout: post
cover: 'Images/nnintrocover.jpeg'
title: Introduction to Neural Networks
date: 2018-08-04
tags: [Neural Networks]
author: Michael Piseno
---

<head>
	<link rel="stylesheet" href="{{ site.baseurl }}styles.css">
</head>

$$\DeclareMathOperator * {\argmax}{arg\,max}$$

> Disclaimer: This tutorial assumes you have a basic understanding of linear algebra (i.e. matrix multiplication and vector operations) and the Python programming language. I will also be using some basic differentiation from calculus, but if you don't understand this then don't worry about it too much. It should still be relatively easy to follow along.

## What This Tutorial Will Teach You

This tutorial will show you how to implement a deep neural network from scratch (i.e. no packages will be used except to import the dataset) to predict handwritten digits. This is the most common problem to solve for beginners learning how neural networks work, but is by far not the only application of deep neural networks, as I'm sure you're aware. You will learn the math behind the network and be able to tweak your own network to learn on other datasets by the end of this tutorial. Typically, when doing machine learning, we will use packages like Tensorflow to do the heavylifting, however as a beginner, its very important that you understand what the network is doing rather than simply calling some abritrary function from a package to do all the work without knowing what's actually going on behind the scenes. 

Additionally, some of you may have heard of more complex types of neural networks like convolutional neural networks or recurrent neural networks. Those have added layers of complexity that we will cover in future tutorials, but for this one we will just focus on the basic deep neural network. Don't worry, it can still learn to recognize handwritten digits pretty well! With that out of the way, lets get started.

## The Problem

If I give you a picture of the number 7, you could easily recognize it as a seven, even if it was presented in different sizes, fonts, positions on the screen, or handwriting. You brain knows that it fundamentally represents the same idea regardless of how the picture is presented. However writing a program that "looks" at an image and tells you what it thinks the image represents is a lot less intuitive. Thats exactly what we are going to try to do today. We are going to write a program that takes in a 28 x 28 pixel greyscale image and outsputs a vector of lengths 10 as its answer. Why length 10? Because there are 10 digits, 0 through 9. Each entry in this output vector will be a probability between 0 and 1 that represents how likely the input image was that index's number. For example, if we give the program an input image of a 7, we hope to get an output that looks something like this:

$$\begin{bmatrix}
    0.01 \\
    0.02 \\
    0.01 \\
    0.005 \\
    0.01 \\
    0.005 \\
    0.01 \\
    0.91 \\
    0.01 \\
    0.01
\end{bmatrix}$$

Notice how index 7 of the previous vector contains the maximum value. This corresponds to the number that the program thinks if the mostly likely of its possible choices. If the program thought the input image was a 0, the maximum value of the output vector would be at index 0. If the program thought it was a 3, the maximum value would be at index 3, and so on.

We could also perform an operation called an argmax on the output vector, to reduce our answer to a scalar value. The argmax function takes a vector as an input and returns the number of the index of the maximum value of that vector. So for example, if we call the vector in the previous example $v$, then $\argmax(v)=7$. This is nice because we can input a picture to our program and it output its prediction as a simple number.

## Artificial Neural Networks

### Structure

An arificial neural network, or ANN for short, is a network of interconnected nodes that together can model a mathematical function. The ANN is arranged in layers of nodes. Think of a node as just a container that holds a number. Each node will take inputs from every node in the previous layer (except the first layer), do some fancy math on its input, and then spit out a output to each node in the next layer. Here's an example of a node taking in an input from each node in the previous layer.

<img class="center-img" src="{{ site.baseurl }}Images/neuron.png">

The first layer, often called the input layer, isn't a real layer and is often just there for show. The last layer is called the output layer. The layers between the input and output layers are called hidden layers. Here's an example of what a network might look like:

<img class="center-img" src="{{ site.baseurl }}Images/network.png">

Additionally, there are weights that lie on the connections between each node. These weights are just numbers that alter the output of one nodes before it is input into its destination in the next layer. Think of the connections as paths between the two nodes that the data must travel on to get to the next layer. While the data is traversing those paths, it gets modified by weights that live on the path. Each path will have its own weight, a scalar value. Then, when the modified data reaches the next layer, it gets input into an "activation function" which lives on each node. Remember, each node in the hidden layers and outut layer takes inputs from EVERY node in the previous layer. We will talk about this function and its purpose in a minute.

So we have an idea of the structure, but how does this relate to our images that we are trying to predict? Well remember each image is 28 x 28 pixels (in this case), or 784 total pixels. Each pixel is represented by a number - its greyscale value. If the number is 0, then the pixel is black. If the number is 1, the pixel is white. If the number of 0.5, the pixel is a greyscale value in between black and white. We will store each image in a vector of length 784 and use that as our input layer. So in essence, the image is being flattened into a vector of greyscale values and is then fed into the network.

The output layer in our case will have 10 nodes, one for each of the possible digits. The nodes in this layer will not output a value to every node in the next layer because there is no next layer. They will simply output one value each, and that value will be the probability value discusses earlier. That means the output of our entire network will be a vector of length 10, and the maximum value in that vector will be our prediction.

As for the hidden layers, choosing how many hidden layers we want and how many nodes are in each hidden layers in beyond the scope of this tutorial and can get quite advances. For now, we'll just choose 1 hidden layer and make it have 15 nodes. So our network will have 3 layers, including the input layer.

<img class="center-img" src="{{ site.baseurl }}Images/network2.png">

### Some Intuition

"Why do we expect this layered structure of nodes to predict anything?", you might be thinking. I had the same thoughts when I was first learning this stuff. We can think of it like this. We want the final layer to tell us which digit it recognizes right? So it might be helpful to think about how we as humans determine what digit we are looking at. For example, we recognize a 9 because it has a loop on top and a line sticking out from the bottom. Similarly, we recognize a 7 as a horizontal line and then a vertical line sticking down on its right side. In other words, we break it down into smaller problems of recognition. So ideally, we want our network's second-to-last layer to be able to recognize and predict smaller components of a digit (e.g. loops or lines or multiple connected lines).

But isn't that just our original problem except scaled down a little bit? Exactly! That's where the hierarchical structure of a neural network comes in. The network breaks down the problem into smaller and smaller problems to get a better prediction. Carrying this idea on, we would ideally want the third-to-last layer to recognize and predict even smaller groups of pixels that make up those loops. Essentially, as we go back in layers, the network has to learn less and less complex structures, and it turns out it can do this pretty well for simple images like greyscale handwritten digits.

So you might think, we don't we just have a million-layer network and predict really complex images? We'll it turns out this is not only really computationally expensive, but there are other hurdles that are more fundamental to the structure of the network that are beyond the scope of this tutorial, but its important to note that neural networks aren't just magic structures that can predict anything - there are definitely limits.

### The Activation Function

Before we get into the specifics of the activation function, we should solidify exactly how data will flow into each node in a more mathematical sense. Lets just examine one node, - we'll call it $q$ - in the second layer of the network (the first hidden layer) and see how it works. As values from the input layer travel along their paths to get to $q$, they run into the weights. The value from each input gets multiplied by its respective weight, and then when they all get to $q$, those weighted valued get summed together. We will call this summed value $z$. In other words,

$$z = w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n} = \sum_{i=0}^{n} w_{i}x_{i}$$

where n is the number of inputs to $q$ and the $x$'s are the greyscale values from the input. Remember the weights are just scalar values. But there are multiple nodes in the hidden layers, so each of these nodes will have an input like the one above. But then what do the nodes do with those inputs? thats where the activation function comes in.

Remember that function that lives on each node in the hidden layers and output layer? The activation function's job is to determine which nodes are "important" and which ones aren't. If the activation function deems the node not important (based on the weighted sum of its imputs), then it will output values that are closer to 0 to all the nodes in the next layer to tell them "hey I'm not important!". If, however, the activation function deems the node important, it will output a large value that scales to its level of importance, and those values are then sent along the paths to the next layer. There are many different types of activation functions, but the ones we will use is a classic one called the sigmoid function which is defined as:

$$sigmoid(z) = \frac{1}{1+\exp(-z)}$$

where $z$ is the same $z$ from above, which is the weighted sum of the inputs to that node.

The sigmoid function maps values onto the open interval $(0, 1)$ (i.e. the function's range is $(0, 1)$). This means that an output of 0 means the node is not important while an output of 1 means the node is very important. To make an analogy to intuition earlier, lets say this particular neuron is responsible for determining if there are any loops in a certain region of this 28 x 28 image. A value of 0 for the output of the sigmoid function would mean that this node is 100% confident that there is no loop in this certain region, while an output of 1 would mean that the node is 100% confident that there IS a loop in that certain region. In reality, we can't expect these level of certainty, so we would probbaly see values that are close to 0 but not exactly 0, or close to 1 but not exactly 1, or somewhere in between.

We can get a visual of the sigmoid function, we can code it up in python:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start=-10, stop=10, num=1000)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

plt.plot(x, sigmoid(x))
plt.show()
```

which produces the following image:

<img class="center-img" src="{{ site.baseurl }}Images/sigmoid.png" width="400" height="300">

In this example we only looked at the domain [-10, 10] but the sigmoid function would still map to the range (0, 1) for all real numbers. Notice that the more positive our input to the sigmoid function (i.e. the more positive our weighted sum), the more confident the mode is in whatever its trying to predict. A more positive input could result from more positive weights in the region that node is focused on coupled with higher greyscale values in that same region. This means that if the weights are very positive around regions where the digit is written, we will get a strong output from our activation function (i.e. positive weights and high greyscale values overlap). Similarly, we will get a very negative output of our activation function if negative weights overlap with high greyscale values. It might be helpful to not only think of weights as scalar that hang out on the connections between nodes, but also visually in the form of a grid of blue and red tiles. The more red the tile, the more negative the weight, and the more blue the tile, the more positive the weight. When we overlap the grid of weights and the image (both 28 x 28), if there is a lot of overlap between blue tiles and the place where the digit is drawn, then we will get a high output value for our activation function for that node. Remember, there will be mutiple of there red-and-blue grids, one for each node in the next layer. Here's an image from the Tensorflow website that captures this idea. Note how the blue areas kind of fit to the area of the digit.

<img class="center-img" src="{{ site.baseurl }}Images/weightvisual.png" width="600" height="300">

How we actually make these weights positive where we want them and negative in others happens when we train the neural network, which will be discussed later.

So to summarize this section, each node in a hidden layer or output layer will take a weighted sum  of input values, put that sum into an activation function, and then spit out a value that tells all the nodes in the next layer how important that node is to whatever its trying to predict.

### Bias

So we've got our nodes to activate based on how positive the weighted sum of their inputs are. But what if we only want our activation function to give strong activations if the weighted sum is higher than some predefined value? Maybe we, or our network, wants to make it difficult to get good activations for certain nodes, and easier for others. This is where the bias term comes in. The bias term shifts the weighted input $z$ down or up, making is more difficult or more easy to get higher outputs from the activation function. Mathematically the bias term for each node will be a $b$, and it comes into play when calculating our activation output. From how on we will use the letter $h$ to denote the activation function output.

$$h = sigmoid(z) = sigmoid(w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n} + b) = sigmoid((\sum_{i=0}^{n} w_{i}x_{i}) + b)$$

If our bias is -10 for example, then our weighted sum has to equal 10 more than before just to get the same activation it had before the bias was introduced, making it more difficult to get higher values of $h$. You can think of the bias term as an extra "node" that sends an input to every node in the next layer, but instead of contributing an $x$ value multiplied by a weight $w$, it simply contributes a $1$ multiplied by the bias value $b$. Visually this would look like the following.

<img class="center-img" src="{{ site.baseurl }}Images/bias.png" width="400" height="200">

Note the +1 node on the bottom of each layer that will be multiplied by the bias term on each of its connections, then added to the weighted sum of each node in the next layer to produce the desired effect. By changing both the weights and the bias during training, we can adjust the "difficulty" of the activation of each node as well as the weights themselves.

