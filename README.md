### Neetcode Machine Learning Practice Solutions


I have spent good time trying to solve the neetcode.io's machine learning questions which I found extremely helpful to remember/understand/learn the foundations of today's machine learning. 
I tried not to use any LLM and tried to just answer the questions in the simplest way possible without caring about the optimization and code quality.
Here I share the solutions (all are accepted solutions) with the people in the field.

## Question 1- Gradient Descent:
## Gradient Descent Minimization

Your task is to minimize the function via Gradient Descent:

**f(x) = x²**

Gradient Descent is an optimization technique widely used in machine learning for training models. It is crucial for minimizing the cost or loss function and finding the optimal parameters of a model.

For the above function, the minimizer is clearly:

**x = 0**

However, you must implement an **iterative approximation algorithm** using **gradient descent**.

### Input Parameters

- **`iterations`**: The number of iterations to perform gradient descent. `iterations >= 0`.
- **`learning_rate`**: The learning rate for gradient descent. `0 < learning_rate < 1`.
- **`init`**: The initial guess for the minimizer. `init != 0`.

### Output

Return the value of `x` that minimizes the function using the given parameters.

> ⚠️ Round your final result to **5 decimal places** using Python's built-in `round()` function.

## Question 2- Linear Regression Forward:
## Linear Regression Implementation

Your task is to implement **linear regression**, a statistical model that forms the foundation of many machine learning techniques, including neural networks.

You must implement two functions:

1. **`get_model_prediction()`** – Returns a prediction value for each dataset value.
2. **`get_error()`** – Calculates the error between predicted and actual values.

---

### 📥 Inputs

#### `get_model_prediction(X, weights)`

- **`X`**: A dataset used by the model to predict the output.
  - `len(X) = n`
  - `len(X[i]) = 3` for `0 <= i < n`
- **`weights`**: The current weights of the model:
  - `w1`, `w2`, and `w3`
  - `len(weights) = 3`

#### `get_error(model_prediction, ground_truth)`

- **`model_prediction`**: The model’s predicted output for each training example.
  - `len(model_prediction) = n`
- **`ground_truth`**: The correct labels/answers for each training example.
  - `len(ground_truth) = n`

---

These functions are essential for training and evaluating a linear regression model, enabling the model to make predictions and assess how far off they are from the true values.

## Question 3- Linear Regression Training
## Training a Linear Regression Model

Now that you've implemented `get_model_prediction()`, it's time to build the training loop for linear regression using **gradient descent**.

You will implement the **`train_model()`** function, which performs parameter updates over a series of iterations.

---

### 🎯 Objective

At each iteration of the training loop:

1. Call `get_model_prediction()` to get current predictions.
2. Call `get_derivative()` to compute the gradients of the error with respect to the weights.
3. Update the weights using gradient descent.

---

### 📥 Inputs

#### `train_model(X, Y, num_iterations, initial_weights)`

- **`X`**: The input dataset.
  - `len(X) = n`
  - `len(X[i]) = 3` for `0 <= i < n`
- **`Y`**: The ground truth labels.
  - `len(Y) = n`
- **`num_iterations`**: The number of iterations to run gradient descent.
  - `num_iterations > 0`
- **`initial_weights`**: Initial weights for the model `[w1, w2, w3]`.
  - `len(initial_weights) = 3`

---

### 📤 Output

- Return the **final weights** after training as a NumPy array with shape `(3,)`.

---

This training function should iteratively improve the model by minimizing the error through gradient descent updates.

## Question 4- Pytorch Basics
**PyTorch** is the industry-standard library for deep learning and was used to train models like ChatGPT. Understanding how to work with PyTorch tensors is essential before building powerful neural networks.

> 🎥 Tip: Check out the first 9 minutes of an introductory video on PyTorch for a summary of basic functions.

---

### 🧠 Your Tasks

1. **Reshape** an `M × N` tensor into a `((M × N) // 2) × 2` tensor.
2. **Calculate the column-wise average** of a tensor.
3. **Concatenate** an `M × N` tensor with an `M × M` tensor into an `M × (M + N)` tensor.
4. **Compute the Mean Squared Error (MSE) Loss** between a prediction and a target tensor.

---

### 📥 Inputs

- **`to_reshape`**: A tensor to be reshaped.
- **`to_avg`**: A tensor from which to compute the average of each column.
- **`cat_one`**: The first tensor to concatenate (shape: `M × N`).
- **`cat_two`**: The second tensor to concatenate (shape: `M × M`).
- **`prediction`**: The output tensor from a model.
- **`target`**: The true label tensor corresponding to the model's input.

---

These exercises will help build a strong foundation in PyTorch tensor operations—an essential step toward implementing deep learning models effectively.

## Question 5- Digit Classifier
Your task is to implement a neural network that can recognize black and white images of handwritten digits. This is a simple but powerful application of neural networks. To learn about coding neural networks in PyTorch, watch this 10 minute clip.

For the model architecture, first use linear layer with 512 neurons follwed by a ReLU activation, as well as a dropout layer with probability p = 0.2 that precedes a final Linear layer with 10 neurons and a sigmoid activation. Each output neuron corresponds to a digit from 0 to 9, where each value is the probability that the input image is the corresponding digit.

### Input:

**images** - one or more 28×28 black and white images of handwritten digits. `len(images) > 0` and `len(images[i]) = 28 * 28` for `0 <= i < len(images)`.

Write the **architecture / constructor** and the **forward()** pass that returns the model's prediction.  
Do not write the training loop (or gradient descent) to minimize the error.

## Question 6- Pytorch Training
## Question 7- Into the Natural Language Processing
In this problem, you will load in a raw body of text and set it up for training. ChatGPT uses the entire text of the internet for training, but in this problem we will use Amazon product reviews and Tweets from X.

Your task is to encode the input dataset of strings as an integer tensor of size `2⋅N×T`, where `T` is the length of the longest string. The lexicographically first word should be represented as 1, the second should be 2, and so on. In the final tensor, list the positive encodings, in order, before the negative encodings.

### Inputs:

- **positive** - a list of strings, each with positive emotion  
- **negative** - a list of strings, each with negative emotion

## Question 8- Sentiment Analysis
Your task is to implement a neural network that can recognize positive or negative emotion in an input sentence. This application of word embeddings is the first step in building ChatGPT. To learn more about word embeddings, check out this video.

The background video is critical to completely understanding the ML concepts involved in this problem.

### Model Architecture

- First use an **embedding layer of size 16**.
- Compute the **average of the embeddings** to remove the time dimension (this is called the **"Bag of Words"** model in NLP).
- End with a **single-neuron linear layer** followed by a **sigmoid activation**.

Implement the **constructor** and **forward()** pass that outputs the model's prediction as a number between **0 and 1** (completely negative vs. completely positive).  
**Do not train the model.**

### Inputs:

- **vocabulary_size** - the number of different words the model should be able to recognize  
- **x** - a list of strings, each with negative emotion

## Question 9- GPT Dataset
Before we can train a transformer like GPT, we need to define the dataset. We take a giant body of text and we can create examples for the model to predict the next token based on different contexts. This is what “ChatGPT was trained on the entire internet” means.

Your task is to write the batch_loader() function which will generate a batch_size * context_length dataset and its labels. Use torch.randint() to pick batch_size different starting words for each sequence.

Inputs:

raw_dataset - a body of text. len(raw_dataset) > 0.  
context_length - how many tokens back the model can read. context_length > 0.  
batch_size - how many sequences to generate. batch_size > 0.  
Return the input dataset X and the labels Y. len(X) = len(Y).

Example 1

Input:  
raw_dataset = "Hello darkness my old friend"  
context_length = 3  
batch_size = 2

Output:  
X = [['darkness', 'my', 'old'], ['hello', 'darkness', 'my']]  
Y = [['my', 'old', 'friend'], ['darkness', 'my', 'old']]

Explanation: The first random index chosen was 1, which became the starting index for the first sequence. Given a context of 'darkness' the model learns that 'my' comes next. Given a context of 'darkness my' the model learns that 'old' comes next. Given a context of 'darkness my old' the model learns that 'friend' comes next. The second random index chosen was 0. Similar reasoning applies for the second sequence.

NOTE: Before feeding this into the model, we would encode each word as an integer, just as done in the NLP Intro problem. It is recommended to solve that problem first.

## Question 10- Self Attention
We're finally ready to code up self-attention. This is the main part of Transformers like ChatGPT. Check out this video for an explanation of the concepts.

The background video is critical to completely understanding the ML concepts involved in this problem. It is a bit lengthy, but is worth the time investment. This problem teaches you, at the lowest level, how LLMs read like humans and focus on what's important. This is definitely the hardest problem in the series.

The class which will be used as a layer in the GPT class just like nn.Linear(). Forward() should return a `(batch_size, context_length, attention_dim)` tensor.

### Inputs:

- **embedding_dim** - the input dimensionality where `embedding_dim > 0`.
- **attention_dim** - the head size where `attention_dim > 0`.
- **embedded** - the input to `forward()`. `embedded_shape = (batch_size, context_length, embedding_dim)`. This tuple format is PyTorch convention for 3-D.

## Question 11- Multi Headed Self Attention
It's time to implement multi-headed self-attention. This layer is what makes LLMs so good at talking like real people. Check out this video for an explanation of the concepts.

Fortunately, this problem is a LOT easier than Self Attention. It's recommended to solve that problem first!

Your task is to code up the **MultiHeadedSelfAttention** class making use of the given **SingleHeadAttention** class. The forward method should return a `(batch_size, context_length, attention_dim)` tensor.

### Inputs:

- **embedding_dim** - the input dimensionality where `embedding_dim > 0`.
- **attention_dim** - the output dimensionality where `attention_dim > 0`.
- **num_heads** - number of self-attention instances where `num_heads > 0` and `attention_dim % num_heads = 0`.
- **embedded** - the input to `forward()` where `embedded.shape = (batch_size, context_length, embedding_dim)`.

## Question 12- Transformer Block
It's finally time to code up the TransformerBlock class. This is the most important class to write in defining the GPT model. It is the gray rectangular box repeated “Nx” times, and it is a giant neural network that uses Multi Headed Attention, among other neural network layers that are given in the starter code.

Note that you should ignore the multi-headed attention and add & norm that comes right before the feedforward block for LLMs. Check out this video for a full explanation.

Your forward() method should return a `(batch_size, context_length, model_dim)` tensor.

## Question 13- Code GPT
We're finally ready to code up the GPT class. This follows the architecture that almost all large language models use, including ChatGPT. Check out this video for a background explanation.

You are given a TransformerBlock, which combines Multi Headed Attention and a FeedForward (Vanilla) Neural Network.

The forward method should return a `(batch_size, context_length, vocab_size)` tensor. The output layer has vocab_size neurons, each representing the likelihood of a particular token coming next.

## Question 14- Make GPT Talk Back
We are going to use our trained model to generate text. Implementing inference is a bit more complex than just calling forward() in a loop. This is because forward() outputs the probability of each possible next character. Given a bunch of probabilities, we have to choose the next token. Check out this video for a background explanation.

The model is also only allowed to read context_length number of tokens back into the past, so we have to cut off characters that are farther back than this threshold at every iteration. Return GPT's response as a String.

### Inputs:

- **model** - an instance of GPT trained on a text file of all Drake songs  
- **new_chars** - the number of tokens GPT should generate  
- **context** - the initial input to the model, typically the newline character  
- **context_length** - the maximum number of tokens the model can read  
- **int_to_char** - a dictionary for decoding numbers back to characters  



