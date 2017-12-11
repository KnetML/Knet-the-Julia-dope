# Knet-the-Julia-dope

## Abstract 

This repo is the [Julia](https://github.com/JuliaLang/julia) *translation* of the [mxnet-the-straight-dope](https://github.com/zackchase/mxnet-the-straight-dope) repo, a collection of notebooks designed to teach deep learning, MXNet, and the `gluon` interface. This project grew out of the MIT course [6.338](http://courses.csail.mit.edu/18.337/2017/) *Modern Numerical Computing with Julia* taught by professor [Alan Edelman](https://github.com/alanedelman). Our main objectives are:

* Introduce the Julia language and its `main` packages in the context of deep learning
* Introduce Julia's package `Knet`: an alternative/complementary option to MXNet
* Leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and code together in one place

Wherever possible we provide Julia's version of any particular code. However, in many instances Julia can achieve the same goal with fewer or different commands such that a direct (or one two one) translation is either impossible or cumbersome. Thus, at our discretion we may replace large blocks of code with a significanly different Julia version, though our objective is to stay true to the essense and goal of the original work. 

In addition to offering a complementary Julia version, wherever possible we also replace or modify examples to experiment with different datasets (with a focus on medical data) for any particular topic (in which case the reader is referred to the original example for context), and we offer additional theoretical explanations whenever approapiate. Finally, with the same spirit of its predecessor, we welcome contributions from the community and hope to coauthor chapters and entire sections with experts and community members. 

## New Datasets

In this repo we implement new examples with different datasets, including:

* Simulated cardiac MRI CINE images
* [Cardiac Electrocardiograms](https://www.physionet.org/physiobank/annotations.shtml)
* [Medical appointment dataset](https://www.kaggle.com/joniarroba/noshowappointments/discussion/28825)

## Implementation with Knet

Throughout this book, we rely upon Julia's package `Knet`. Knet relies on the [AutoGrad](https://github.com/denizyuret/AutoGrad.jl) package and the [KnetArray](http://denizyuret.github.io/Knet.jl/latest/reference.html#KnetArray-1) data type for its functionality and performance. AutoGrad computes the gradient of Julia functions and KnetArray implements high performance GPU arrays with custom memory management.

We make heavy use of packages like [JuliaDB](https://github.com/JuliaComputing/JuliaDB.jl), [Plots](https://github.com/JuliaPlots/Plots.jl) and the plotting backend [GR](https://github.com/jheinen/GR.jl),[HDF5](https://github.com/JuliaIO/HDF5.jl), [Distributions](https://github.com/JuliaStats/Distributions.jl), and ofcourse [Knet](https://github.com/denizyuret/Knet.jl)

## Dependencies 

To run these notebooks, you'll want to install [Julia](https://github.com/JuliaLang/julia) and add all the required packages (this is done automatically for you at the start of every notebook) that we will use throughout this tutorial. Fortunately, after installing Julia this is very easy with the command `Pkg.add('PackageName')`. You'll also want to install [IJulia](https://github.com/JuliaLang/IJulia.jl), a Julia kernel for Jupyter. 

## Table of contents

### Part 1: Deep Learning Fundamentals
* **Chapter 1:** Crash course
    * [Preface](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter01_crashcourse/preface.ipynb)
    * [Introduction](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter01_crashcourse/introduction.ipynb)
    * [Manipulating data with KnetArray](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter01_crashcourse/KnetArray.ipynb)
    * [Linear algebra](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter01_crashcourse/linear-algebra.ipynb)
    * [Probability and statistics](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter01_crashcourse/probability.ipynb)
    * [Automatic differentiation via ``autograd``](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter01_crashcourse/autograd.ipynb)
    
 * **Chapter 2:** Introduction to supervised learning
     * [introduction](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section1-introduction.ipynb)
     * [linear-regression](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section2-linear-regression.ipynb)
     * [logistic-regression](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section3-logistic-regression.ipynb)
     * [multiclass-logistic-regression](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section4-multiclass-logistic-regression.ipynb)
     * [regularization](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section5-regularization.ipynb)
     * [perceptron](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section6-perceptron.ipynb)
     * [learning-environments](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter02_supervised-learning/section7-learning-environments.ipynb)    
     
  * **Chapter 3:** Deep neural networks (DNNs)
      * [multilayer perceptrons](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter03_deep-neural-networks/section2-multilayer%20perceptrons.ipynb)
      * [dropout](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter03_deep-neural-networks/section3-dropout.ipynb)
  
  * **Chapter 4:** Convolutional neural networks (CNNs)
      * [cnn](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter04_convolutional-neural-networks/section2-cnn.ipynb)
  
  * **Chapter 5:** Recurrent neural networks (RNNs)
      * [MLP](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter05_recurrent-neural-networks/section2-MLP.ipynb)
      * [LSTM](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter05_recurrent-neural-networks/section3-LSTM_shakespeare.ipynb)

### Part 2: Applications

  * **Chapter 12:** Time series
      * [Exponential Smoothing and Innovation State Space Model](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter12_time-series/section2-issm.ipynb)
  
  * **Chapter 14:** Generative adversarial networks
      * [introduction](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter14_generative-adversarial-networks/section1-introduction.ipynb)
      * [GANs](https://github.com/moralesq/Knet-the-Julia-dope/blob/master/chapter14_generative-adversarial-networks/section2-generative-adversarial-networks.ipynb)
  