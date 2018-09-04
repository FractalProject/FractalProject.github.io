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

Here's how we will approach this problem. We have our dataset of images, which will probably contain tens of thousands of images. We will split this dataset up into a training dataset and a testing dataset. Once we have our neural network, we will use the training dataset, which is composed of training examples (each image) to train the network and make it learn how to properly recognize digits. Then once we are satisfied with how much we've trained the network, we will test it on the testing dataset. The purpose of testing on a separate dataset than you trained on is to see how well your network generalizes to examples it hasn't seen before. Its very important that you don't use the testing dataset at all until the very end, when you're conpletely done training the network, otherwise we can't be sure that the network actually generalizes well if it gets good results on the testing dataset. The choice of how to divide up the main dataset into training and testing is up to you. It's common to do 90% training and 10% testing, or even 50% training and 50% testing. For our purposes, its not too important. For brevity, I'll refer to the training dataset and testing dataset as "train set" and "test set", respectively. 

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

So we've got our nodes to activate based on how positive the weighted sum of their inputs are. But what if we only want our activation function to give strong activations if the weighted sum is higher than some predefined value? Maybe we, or our network, wants to make it difficult to get good activations for certain nodes, and easier for others. This is where the bias term comes in. The bias term shifts the weighted input $z$ down or up, making it more difficult or more easy to get higher outputs from the activation function. Mathematically the bias term for each node will be represented by a $b$, and it comes into play when calculating our activation output. From now on we will use the letter $h$ to denote the activation function output.

$$h = sigmoid(z) = sigmoid(w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n} + b) = sigmoid((\sum_{i=0}^{n} w_{i}x_{i}) + b)$$

If our bias is -10 for example, then our weighted sum has to equal 10 more than before just to get the same activation it had before the bias was introduced, making it more difficult to get higher values of $h$. You can think of the bias term as an extra "node" that sends an input to every node in the next layer, but instead of contributing an $x$ value multiplied by a weight $w$, it simply contributes a $1$ multiplied by the bias value $b$. Visually this would look like the following.

<img class="center-img" src="{{ site.baseurl }}Images/bias.png" width="400" height="200">

Its important to note that there may be a different bias value for EACH node in a particular layer. The calculation of $h$ above only calculates the output of one node, but each node in a layer will have its own calculation, so there isn't just one bias value. Later we will represent the bias as a vector that holds the biases for each node in a particulat layer. By changing both the weights and the bias during training, we can adjust the "difficulty" of the activation of each node as well as the weights themselves.

Now, just to be completely clear, lets talk about how many weights and biases we have in our network. Our network have three layers. The first layer (input layer) has 784 nodes, the second layer (hidden layer) has 15 nodes, and the output layer has 10 nodes. Every node in the input layer is connected to every node in the hidden layer, a total of $784 \times 15 = 11760$ connections from the input layer to the hidden layer. Each one of these connections will have its own weight. Additionally, since each node in the hidden layer will require its own bias for calculating its $h$ value, we will need 15 bias values. By the same logic, the number of weights and biases needed for the hidden layer to the output layer is $15 \times 10 = 150$ and 10 respectively. All together our network has 11935 trainable parameters. 11935 dials we can tweak to model a function to predict handwritten digits! When we say a neural network "learns", we mean the process by which it adjusts those trainable parameters to get better at doing some task, which in this case is modelling handwritten digits.

### Notation

For some weight $w_{i,j}^{(l)}$ being multiplied by some $h_{j}$, $l$ is the layer containing the $h$ values that the weights are being multiples by. $i$ is the node number of the node in layer $l+1$ that this weight is being sent to (i.e. the node at the end of the road that this weight lives on). $j$ refers to the node number in layer $l$ that this weight comes from, and also the node number that contains $h_{j}$ (i.e. the node at the beginning of the road that the weight lives on). Note that the $(l)$ does not mean the $l$th power, but is simply a superscript to denote the layer number.

For some $b_{i}^{(l)}$, $i$ refers to the node number that the bias is connected to in layer $l+1$, while $l$ refers to the layer that the bias live on.

For some $h_{j}^{(l)}$, $j$ denotes the node number in layer $l$ of the network. For exmaple, the outut of the fourth node in the input layer is represented as $h_{4}^{(1)}$, and the output of the thirteenth node in the hidden layer is:

$$h_{13}^{(2)} = sigmoid((\sum_{j=0}^{n} w_{13,j}^{(1)}x_{13,j}^{(1)}) + b_{13})$$

Lets put the above expression into words. To get the output of the thirteenth node in layer 2 (the hidden layer), we take the sum over all the nodes in the previous layer (layer 1) of the output values of those nodes multiplied by their corresponding weight, and then add the bias term for node 13. Then we apply the activation function and get our desired output.

Having seemingly complicated notation like this will actually make it easy to represent the values in our network as matrices and vectors, and will also allow us to code our network much more efficiently.

### Vectorization

Recall that $z$ was the weighted sum of each node in a particular layer, including the bias term, and was the input of out sigmoid function to that $h = sigmoid(z)$. Previously, calculating every $z$ for a particular layer was pretty involved, but now using vectorization we can simplify the calculations into an easy matrix multiplication.

Lets define $W^{(l)}$ as the matrix containing all the weights for a particular layer $l$. And we'll let $h^{(l)}$ be the vector that gets multiplied by those weights, and $b^{(l)}$ as the bias vector. These are vectors because they contain the values for EVERY node in layer $l$, not just one. This will let us calculate the vector $h^{(l+1)}$, the output of the entire next layer.

$$W^{(l)} = \begin{bmatrix}
    w_{1,1}^{(l)} & w_{1,2}^{(l)} & \cdots & w_{1,n}^{(l)} \\
    w_{2,1}^{(l)} & w_{2,2}^{(l)} & \cdots & w_{2,n}^{(l)} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{k,1}^{(l)} & w_{k,2}^{(l)} & \cdots & w_{k,n}^{(l)} \\
\end{bmatrix}\hspace{1cm}

h^{(l)} = \begin{bmatrix}
    h_{1}^{(l)} \\
    h_{2}^{(l)} \\
    \vdots \\
    h_{n}^{(l)} \\
\end{bmatrix}\hspace{1cm}

b^{(l)} = \begin{bmatrix}
    b_{1}^{(l)} \\
    b_{2}^{(l)} \\
    \vdots \\
    b_{k}^{(l)} \\
\end{bmatrix}$$

We can easily calculate the output of the entire layer $l+1$ by computing the following:

$$h^{(l+1)} = sigmoid(z^{(l+1)}) = sigmoid(W^{(l)}h^{(l)} + b^{(l)})$$

Look carefully at whats going on in this matrix multiplication. If you remember from linear algebra, the first entry in $z^{(l+1)}$ will be $w_{1,1}^{(l)}h_{1}^{(l)} + w_{1,2}^{(l)}h_{2}^{(l)} + w_{1,n}^{(l)}h_{3}^{(l)} + b_{1}^{(l)}$, which is exactly what we had before from calculating the weighted sum for the first node in layer $l+1$. If we carry out the complete matrix multiplication, we get a $z$ vector with $k$ terms, where $k$ is the number of nodes in layer $l+1$. All we're doing is taking the dot product of one row of $W^{(l)}$ with $h^{(l)}$ and adding the bias to get our weighted sum, and then doing that $k$ times - once for each of the nodes in layer $l+1$ so that our resulting vector has one entry for each output of a node in layer $l+1$. This is how we get the output of the entire layer with one simple matrix calculation instead of a bunch of summations. Using NumPy, we'll be able to do these matrix multiplicatons extremely fast in Python.

### A Note

So now we know how to take an input (in our case in the form of 784 greyscale pixel values), and propogated those values through the network using vectorization, multiplying the appropriate weights and adding the appropriate biases when necessary to eventually get our length 10 vector output.

By now hopefully how can see how a neural network can model a really complicated function. And thats essentially all we're trying to do - approximate a function that maps inputs in the form of images to outputs in the form of a predicted digit. Our function takes 784 inputs, produces 10 outputs, and has 11935 parameters that we will tweak, but its still nonetheless a function just like $y=mx+b$ is a function with 1 input, 1 output, and 2 parameters.

In the next section we will see how to actually tweak our parameters so that in the final section we can translate it all into python code and get a working neural network!

## Training the Network

In our dataset we have thousands of examples of handwritten digits and each one of them have a label of the actual digit its supposed to represent. The first of these training examples from our dataset might look like $(x_{1},y_{1})$, where $x_{1}$ is the 28 x 28 pixel image and $y_{1}$ is the number 7, which is the true value of what that image represents. We will use an algorithm that runs the 28 x 28 image through our network, checks the maxmimum of the length 10 vector output to see what prediction the network gave our image, and then compare it to the truth value. We will do this for all our training examples (nont just the first one) and see how accurate the network is, and then adjust the parameters to minimize a value called the "loss", which I'll introduce in a minute.

### The Cost Function

So we have this idea of weights that live on connections between nodes and the value of the weight is annalogous to the strength of that connection. We also have biases, which make it harder or easier for a particular node to give higher outputs from its activation function. At first, we will randomly intiallize these weights and biases, since if we had an idea of which weights and biases to choose already, we really wouldn't need to traing the network would we? It goes without saying that using random parameters will make our network perform horribly at first when we run our images through it, however after some analysis and tweaking of the parameters, we will start to see which direction we need to tune them to make the network more accurate.

How do we measure how good our network performed? When we first randomly initialize the parameters of the network, our output vector will look like a mess, and likely somewhat evenly distributed between in its values (i.e. there won't be a really clear maximum). We need a way to give these types of outputs a "bad score" and give a "good score" to the types of outputs that have a large value in the correct position of the output vector with a low value in incorrect positions. Thats where the cost function comes in.

There are many different funtions you could use for the cost, but one of the simplest is the square error. Basically we will subtract our output vector from the truth vector and square those differences, then add up the entire vector we get the overall cost. For an image whose truth value is 3, that might look like this:

$$sum\left(\left(\begin{bmatrix}
    0.05 \\
    0.05 \\
    0.10 \\
    0.05 \\
    0.10 \\
    0.05 \\
    0.15 \\
    0.15 \\
    0.10 \\
    0.20
\end{bmatrix}
-
\begin{bmatrix}
    0.00 \\
    0.00 \\
    0.00 \\
    1.00 \\
    0.00 \\
    0.00 \\
    0.00 \\
    0.00 \\
    0.00 \\
    0.00
\end{bmatrix}\right)^2\right)$$

Note that the square operation will be applied element-wise to the vector. By using this cost function, we can get a scalar value of how bad out network performed in predicting this digit. What we will end up doing is applying this cost function to every training example and then taking the average cost across all training examples to evaluate how well our network performs. So out final cost function will be defined as follows:

$$J(w, b) = \frac{1}{m}\sum_{n=1}^{m}(y_{n, truth} - y_{n, pred})^2$$

Where $J$ is the cost, $m$ is the number of training examples in our train set, $y_{n, truth}$ is the $n$th training label, and $y_{n, pred}$ is the neural network's predication vector for training example $n$. $w$ and $b$ just signify that this function takes all the weights and biases as input so that it can calculate the output of the network.

Its important to distinguish between the function that the neural network is trying to approximate from the cost function - They are not the same thing! The former, in our case, is a mapping from 28 x 28 images to a digit number, whereas the latter is a mapping from our parameters to a scalar value that evaluates our network's performance. 

### Gradient Descent

So we have some sort of measure of how bad our network is performing, but what do we do about it? We need to find some way of adjusting the weights and biases to that our network performs better. How do we know when the network performs better? Well our cost function is the square difference between the predicted digit from our and the actual 10-dimensional vector representation of the digit. So if we minimize the output of the cost function, this amounts to more closely predicting the correct digit. Now we have a direction - we must solve a minimization problem. Essentially what we will be doing is "descending" the gradient of the cost function to get to a minimum value (hence the name gradient descent for this algorithm of adjusting network weights and biases).

The problem is that this is no simple minimization problem. The cost function can exist is very high dimensional space with many local minima which pose a problem for the gradient descent algorithm. The algorithm tends to get "stuck" in these local minima and unable to jump out and find a global minumum, but often these local minima can still provide good results when evaluating our network on our test set.

<img class="center-img" src="{{ site.baseurl }}Images/gradientdescent.png" width="300" height="200">

Take a look at the above image of a cost versus weight graph that will give a simple one-dimensional gradient descent example. This is not our digit classification exmaple, but a simple gradient descent exmaple to help explain the process easier. On the horizontal axis we have some weight $w_{1}$ and on the vertical axis we have our cost $J$. We start out with a randomly initialize weight and we evaluate our cost, which is high. So we adjust the weight and evaluate the cost again, and do this repeatedly until we converge on a minimum value. We will do this in the following way:

$$w_{1, new} = w_{1, old} - \alpha\frac{\partial}{\partial w_{1, old}}J(w)$$

The partial derivative of the cost funtion with respect to the weight tells us the direction of greatest increase, and so taking the negative of that value will tell u the direction of greatest decrease (remember we want to decrease the cost as much as possible). $\alpha$ is a value called the learning rate, which bascially tells us how far we want to step over when we adjust our weight. If $\alpha$ is too big though, we might overshoot our targeted minimum value, so we usually keep $\alpha$ small. The value of $\alpha$ is defined by the programmer explicitly. The above formula is just taking our old weight, and moving is slightly over in the direction of greatest decrease, i.e. more toward the minimum value for the cost function.

But our problem doesn't have just one weight, it has thousands. So we must adjust all our weights in such a way that the entire cost function takes a step toward its minimum value. To do that, we define a huge vector $\overline{W}$ (read, "W bar") that contains every weight and bias in the entire network. All 11935 of them. Our cost function is a function of all of these weights and biases, so it exists in 11935 dimentional space. Borrowing some knowledge from multivariable calulus, taking the negative gradient of the cost function, then, will give us the direction of greatest decrease:

$$\overline{W}_{new} = \overline{W}_{old} - \alpha\nabla J(w, b)$$

Whats really happening in the above equations it that each weight and bias is getting updated as follows:

$$w_{i,j, new}^{(l)} = w_{i,j, old}^{(l)} - \alpha\frac{\partial}{\partial w_{i,j, old}^{(l)}}J(w, b)$$

$$b_{i, new}^{(l)} = b_{i, old}^{(l)} - \alpha\frac{\partial}{\partial b_{i, old}^{(l)}}J(w, b)$$

This may look intimidating if you're not used to seeing multivariable calculus in an applied setting, but its actually a lot easier in practice than it may at first seem. In the next section, I'll explain exactly HOW we compute this gradient through a method called backpropagation. Then, I'll do a short recap of how everything fits together, before finally getting to the coding part of this tutorial.

### Backpropagation

In this section, I'll explain how we compute the gradient part of the gradient descent algorithm (the $-\nabla J(w, b)$ part) so we know which direction to change the weights. The method of doing this is called backpropagation, or backprop for short. When we compute this gradient vector $-\nabla J(w, b)$, it contains values that tell you the direction of the greatest decrease for each dimension in the parameter space (i.e. n-dimensional space created by the parameters of our network that the cost function exists in). For example, this might be the value of your gradient vector for some particular input parameters to the cost funtion:

$$\begin{bmatrix}
    0.27 \\
    -0.14 \\
    2.56 \\
    \vdots \\
    0.95 \\
    -1.80
\end{bmatrix}$$

Just like we might have a vector $(1, 3)$ that specifies a direction in 2-dimensional space, our gradient vector tells us a direction in n-dimensional space, where n is the number of parameters in the network, and this direction is the direction of greatest decrease for the cost function. Remember, our partial derivitives are with respect to the weights and biases, but the derivatives themselves are performed ON the cost function. The values in this gradient vector also tell us how "sensitive" our parameters are to changes. For example, lets extract two arbitrary values from this gradient vector, say 0.2 and 1.0. And let say, for the sake fo argument, that the 0.2 is the partial derivtive of the cost function with repsect to the weight that connects node 1 of the input layer to node 12 of the second layer. And the 1.0 corresponds to the partial derivative of the cost function with respect to the weight that connects node 5 of the input layer to node 7 of the second layer. To put this in mathy-terms, this is what I'm talking about:

$$0.2 = \frac{\partial}{\partial w_{12, 1}^{(1)}}J(w, b)$$

$$1.0 = \frac{\partial}{\partial w_{7, 5}^{(1)}}J(w, b)$$

That means weight $w_{7, 5}^{(1)}$ is 5 times more sensative to changes than weight $w_{12, 1}^{(1)}$, since its gradient was 5 times as big. You can also think of it as the slope being 5 times as steep in the direction of the parameter space corresponding to weight $w_{7, 5}^{(1)}$ then the direction corresponding to weight $w_{12, 1}^{(1)}$.


