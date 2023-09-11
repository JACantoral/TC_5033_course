# Advanced Machine Learning Methods (Deep Learning)


# Very Detailed Syllabus

# Module 1: Introduction to Deep Learning and Fully Connected Networks (Weeks 1-2)

In this first module, you will explore the foundational principles of Deep Learning to understand the underpinnings of Fully Connected Networks. This module demystifies the structure of neural networks, the significance of layers and nodes, and the role they play in building intelligent systems. You will also have a solid understanding of the essential theory behind backpropagation, while appreciating its vital role in refining neural network predictions.

### Learning objectives
1. Understand the basic principles of artificial intelligence, machine learning, and deep learning, and distinguish among them.
2. Comprehend the structure and operation of a simple neural network, including understanding the role of activation functions and the process of backpropagation for learning network weights.
3. Implement a multilayer fully connected network, adjust its hyperparameters, and evaluate its performance on a given dataset.

## Week 1: Introduction to Deep Learning
#### Introduction:

In the first week, you will get a foundational understanding of what neural networks are. The focus will be on basic elements like neurons, activation functions, and feedforward mechanisms.
#### 1.1 First Active Class (Thu 14 Sept 18 hrs)
	    - Introduction to Machine Learning, Artificial Intelligence, and Deep Learning. A brief History.
	    - Comparison of classical algorithms vs. deep learning
	    - Types of Machine Learning: Supervised and Unsupervised
	    - Neural Networks: Perceptron, Multi-layer Perceptron
	    - Activation Functions: Sigmoid, Tanh, ReLU.
#### 1.2 Home Activities
	- Watch the following videos prior to session 2
	    - https://youtu.be/jGNYsjHTnho
	    - https://youtu.be/XoIj-omJAZo
	    - https://youtu.be/3KhumV1S6vs
	    - https://youtu.be/ie-tCP7YYrI
	- Reading:
		- Chapter 11 from the Book
		- Paper 1, and paper 2
		
## Week 2: Fully Connected Networks
#### **Introduction:**  
Week 2 dives into fully connected neural networks, also known as dense layers. The main highlight will be the backpropagation algorithm, the cornerstone for training neural networks.
#### 2.1 Second Active Class (Mon 18 Sept 18:30 hrs and so on)
    - Detailed walkthrough of a fully connected network (also known as Dense or Feedforward networks)
    - Forward propagation & backward propagation in depth
    - Cost functions: Cross Entropy for multi-class problems
    - Regularization and improvements: Dropout, Batch Normalization

#### 2.2 Home Activities
	- Watch the following videos:
		- https://youtu.be/lnDjwepC-5I
		- https://youtu.be/Cr5cYDrMYZQ
		- https://youtu.be/4nDsiV3GXa8
		- https://youtu.be/qQELiV1_GHA   PyTorch tutorial

#### 2.3 Activity 1 release (due on Week 3)


# Module 2: Convolutional Neural Networks (Weeks 3-4)
In this module you will take a journey into the visual cortex of Deep Learning – the Convolutional Neural Network (CNN). Discover how CNNs have revolutionized image recognition by emulating human visual perception. Explore seminal architectures such as AlexNet, VGGNet, and ResNet, and delve into techniques like Batch Normalization and He Initialization to enhance model performance.

### Learning objectives
 1. Understand the structure and functionality of Convolutional Neural Networks (CNNs), including concepts like convolution, pooling, and feature (activation) maps.
2. Get to know famous architectures like LeNet, AlexNet, VGG16, ResNet, and Inception, and appreciate how CNN architecture has evolved.
3. Apply a CNN to a real-world image classification problem, leveraging transfer learning techniques.

## Week 3: Introduction to Convolutional Networks
#### Introduction

Week 3 introduces the basics of Convolutional Neural Networks (CNNs). Students will learn how to handle image data and understand the theory behind convolutions.

    - Image representation, Image Processing Fundamentals
    - Convolution Operation, Pooling, Stride, Padding
    - Architecture of CNN: Convolution Layers, Pooling Layers, Fully Connected Layers
    - Case Study: LeNet
    
#### 3.1  Home Activities
		- Videos to watch:
			- https://youtu.be/cUa3Jug3TiA
			- https://youtu.be/xkZD5eB5KVM
			- https://youtu.be/XkqgTaWle0I
			- https://youtu.be/w9ECvUxMAJQ
		- Readings:
			- Book Chapter 14
			- Papers 1 and 2 from suggested readings Module 2

#### 3.2 Activity 1 due this week


## Week 4: Advanced Convolutional Networks

### Introduction:
In Week 4, students will explore more complex CNN architectures. They will learn about techniques like pooling, batch normalization, and different CNN architectures like LeNet, AlexNet, etc.

#### 4.1 Third Active Class
	- Question on previos material
    - Advanced CNN architectures: AlexNet, VGG16, ResNet, Inception
    - Transfer learning with pre-trained networks
    - Object detection, segmentation: R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN

#### 4.2 Home Activities
	- Readings:
		- Papers 3 to 6 from suggested readings in Module 2
	

#### 4.3 Activity 2 release  (due on week 5)

# Module 3: Recurrent Neural Networks (Weeks 5-6)
In this module you will explore sequential data processing with Recurrent Neural Networks (RNNs). RNNs, including their powerful variants LSTMs and GRUs, enable machines to understand and generate sequences, making them indispensable for tasks like natural language processing, time-series forecasting, and more. Understand their architecture, applications, and the challenges they address in maintaining and processing temporal information.

### Learning objectives
1. Understand the structure and functionality of Recurrent Neural Networks (RNNs), including how they handle sequence data and the problems of vanishing and exploding gradients.
2. Learn about more complex recurrent structures like LSTMs and GRUs, and understand when and why they're used.
3. Implement a recurrent neural network for a sequence prediction task, such as text generation or time-series forecasting.

## Week 5: Introduction to Recurrent Neural Networks

#### Introduction:
Week 5 focuses on Recurrent Neural Networks (RNNs), which are ideal for handling sequential data. The architecture and basic RNN cells will be the primary focus.

    - Sequence data and its importance
    - RNNs and their variants: Unrolled RNN, Bidirectional RNN
    - Problems in RNNs: Vanishing and Exploding gradients
    - Case Study: Character level language model

#### 5.1 Home Activities:
	Reading:
		- Book Chapter 15
		- Papers 1, 2, from module 3
### 5.2 Activity 2 due on this week

## Week 6: Advanced Recurrent Networks and Language Models

#### Introduction:
Week 6 builds on the previous week's knowledge, diving deeper into specialized RNN architectures like Long Short-Term Memory (LSTM) units and Gated Recurrent Units (GRUs).

    - Long Short Term Memory (LSTM) networks
    - Gated Recurrent Units (GRUs)
    - Text generation, Text classification
    - Case Study: Text generation using LSTMs

#### 6.1 Home activities:
	- Readings:
		- Papers 3, 4 from module 3 list
#### 6.2 Activity 3 release (due on week 7)

#### 6.3 Quiz 1 W1-W5
	

# Module 4: Attention Mechanisms and Transformers (Weeks 7-8)
In this module you will study one of the most important advancements in recent Deep Learning: Attention Mechanisms and the Transformer architecture. Discover how these innovations have overcome the limitations of traditional neural network designs to achieve unparalleled performance in tasks demanding context awareness and positional sensitivity, particularly in language translation and understanding

### Learning objectives
1. Understand the concept of attention in deep learning, its necessity, and its impact on model performance, particularly in sequence-to-sequence tasks.
2. Understand the structure and functionality of Transformer models, including the concept of self-attention and position encoding.
3. Implement a transformer model for a complex task like machine translation or fine-tune a pre-trained transformer for a specific task.

### Week 7: Attention Mechanisms

#### Introduction:
Week 7 shifts the focus to attention mechanisms in neural networks. These are crucial components for handling sequences and have been revolutionary in NLP. Students will understand how attention works and why it is a powerful feature in modern neural architectures.

#### 7.1 Fourth Active Class (October 27th 17:00hrs)
    - The problem with Seq2Seq Models
    - Introduction to Attention Mechanisms
    - Types of Attention: Additive attention (Bahdanau), Multiplicative attention (Luong)
    - Case Study: Neural Machine Translation with Attention
#### 7.2 Home activities
	- Readings:
		- Book Chapter 16
		- Paper 1 from Module 4
#### 7.3 Activity 3 due on this week
	
### Week 8: Introduction to Transformers

#### Introduction
Week 8 introduces the Transformer architecture, which has been seminal in various machine learning applications, especially in natural language processing. You'll delve into the nuts and bolts of this architecture, including self-attention and positional encoding.

    - Limitations of RNNs and the need for Transformers
    - Architecture of Transformers: Self-attention, Positional Encoding
    - Case Study: OpenAI's GPT and Google's BERT

#### 8.1 Home activities:
	- Readings:
		- Paper 1, 2, 3, from module 4

#### 8.2 Activity 4 and activity 5 release


## Module 5: Advanced Topics and Trends in Deep Learning (Weeks 9-10)

This module provides an introduction to advanced topics that are shaping the future of AI. You will study the mechanics and applications of Variational Autoencoders and Generative Adversarial Networks. Understand their potential in tasks like data generation, anomaly detection, and image synthesis.

### Learning objectives
1. Understand the concepts behind different generative models, including Autoencoders, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs).
2. Stay updated with recent developments and future trends in Deep Learning, including self-supervised learning, meta-learning, and the ethical considerations surrounding AI.
3. Apply a generative model to a real-world problem, like image generation or anomaly detection, and critically reflect on its performance.

### Week 9: Generative Models
#### Introduction
Week 9 moves into the exciting realm of generative models. These types of models have the capability to generate new, unseen data based on what they've learned. You'll explore generative models like GANs and VAEs, understanding their architecture and applications.
## 9.1 Fifth Active class (November 13th 17:00hrs)

	- Introduction to Generative Models
    - Autoencoders
    - Variational Autoencoders (VAEs)
    - Generative Adversarial Networks (GANs)
    - Case Study: Image Generation with GANs

#### 9.1 Home Activities:
	- Readings:
		- Papers 1-3 from Module 5
#### 9.2 Activity 4 due on this week



### Week 10: Recent Developments and Future Directions

#### Introduction:
The final week, Week 10, looks toward the future. You'll discuss emerging trends, technologies, and challenges in the deep learning landscape. This week serves as a horizon-expanding session that paves the way for your continued learning in the field.

### 10.1 
    - Review of course and trends in Deep Learning
    - Self-supervised learning, Meta-learning, Few-shot learning
    - Ethics in AI and Deep Learning: Bias, Fairness, Interpretability
    - Discussion on the future of Deep Learning, Open Research Problems

#### 10.2 Activity  5 due on this week
#### 10.3 Home activities:
	- Readings:
		- Papers 4-6 from Module 5

