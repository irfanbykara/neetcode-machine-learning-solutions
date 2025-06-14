### Neetcode Machine Learning Practice Solutions


I have spent good time trying to solve the neetcode.io's machine learning questions which I found extremely helpful to remember/understand/learn the foundations of today's machine learning. 
I tried not to use any LLM and tried to just answer the questions in the simplest way possible without caring about the optimization and code quality.
Here I share the solutions (all are accepted solutions) with the people in the field.

## Question 1- Gradient Descent:
Your task is to minimize the function via Gradient Descent: 
f
(
x
)
=
x
2
f(x)=x 
2
 .

Gradient Descent is an optimization technique widely used in machine learning for training models. It is crucial for minimizing the cost or loss function and finding the optimal parameters of a model.

For the above function the minimizer is clearly x = 0, but you must implement an iterative approximation algorithm, through gradient descent.

Input:

iterations - the number of iterations to perform gradient descent. iterations >= 0.
learning_rate - the learning rate for gradient descent. 1 > learning_rate > 0.
init - the initial guess for the minimizer. init != 0.
Given the number of iterations to perform gradient descent, the learning rate, and an initial guess, return the value of x that globally minimizes this function.

Round your final result to 5 decimal places using Python's round() function.

