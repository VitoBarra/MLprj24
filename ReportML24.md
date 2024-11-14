# Implementation MLP Project Requirement
1. backpropagation
   1. Add Momentum
   2. Add Regularization L2
2. Try the monk Database
   1. hyperparameter search.
      1. Plot Learning curve
   2. results on internal test set
   3. heuristic to use
      1. Start from different starting point
      2. early stopping
   4. Implement some validation technique, Cross-validation or Old-Out
      



## Implementation of MLP
What is Multy Layer Perceptron? It is the simplest form of a feed-forward neural network. It is composed of multiple neurons.
Each neuron is a computation unit and it is defined as:
$$o_j=f_{\sigma}\left(\sum^n_{i=0} w_{i,j}o_i\right)$$


Having multiple neurons means that we want to be able to calculate $o_i$ which is the output of the neurons in the previous layer.
previes
We have to start from the input. For each input $x_i$ we will have a first layer where $$o_i=x_i \ \ \forall i=0,...,n_{in}$$ where $n_{in}$ is the dimensionality of the input $\mathbf{x}$.

The second layer will be $$o_j=f_{\sigma}\left(\sum^n_{i=0} w_{i,j}o_i\right) \forall j=0...m_1$$ where $m_1$ is the number of neuron of that layer.

The output layer will be $$o_j=f_{out}\left(\sum^n_{i=0} w_{i,j}o_i\right) \forall j=0...n_o$$ where $n_o$ is the dimensionality of the output.

In this way we can have a 3 layer Feed-Forward, but it can be expanded to add an arbitrary number of layers.



# Implement Backpropagation.
The backpropagation is a generalization of the $\delta$ rule with multi layer perceptron






# Learing strategy
We have to implement each of the possible learning strategy
-  Batch: each the pattern $p$ are presented to the network and the new gradient is evaluated to update the weight
-  Stocastic/on-line: a single pattern $p$ is passed through the network and the weight is update before going to te next patter 
-  Mini-Batch: a set of $k$ patter are passed to the network and the gradient in evaluated on the $k$ patter and the weight are calculated

__batch__ is considered to be the most stable, but it is also the slowest because each patern need to be loaded in memory e usualy they dont fit in the GPU memory

__Stocastic/On-line__ is the fasts in terms of performance, but it is noisy and unstable

__Mini-Batch__: Is usually the go to choose since offer a good stability and a good efficiency 


# Add momentum
The momentum is a technique to have a more stable gradient descent iteration.

it is done by changing the update rule where instead of just changing the with using the gradient 
you also add the old gradient
so instead of having $$\mathbf{w}_{new}=\mathbf{w} -\eta \nabla W$$
we have 
$$W_{new}= \mathbf{w} -\eta\nabla \mathbf{w} + \alpha\nabla W_{old}$$

the ensure stability by dumping the oscillation on the gradient

#### Nesterov momentum variant
the Nesterov Variant is the following:
first add the momentum to the current with and save this a temporary with $W_{temp}$ and then calculate the gradiant at this temporary point
the update rule become as follows:
$$\begin{array}{}
\mathbf{w}_{temp} &=& \alpha \mathbf{w} +   \eta\nabla W_{old}\\
\mathbf{w}_{new} &=& \mathbf{w} + \nabla W_{temp}
\end{array}$$
This update rule is know to have a faster convergence, used in batch mode and [maybe for SGD (mini-batch)](https://medium.com/@giorgio.martinez1926/nesterov-momentum-explained-with-examples-in-tensorflow-and-pytorch-4673dbf21998)

[Da capire meglio questa fonte](https://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-acc)

# Add Regularization L2
L2 is for the Norm 2.
the regularization is a method that is used to control the complexity of the model in order to reduce overfitting and improve generalization. 
L2 is the common Tikhonov regularization that add a penalty term to the loss. This penalizes large weights values and encourage the model to prefer smaller values to reduce the complexity of the model itself

The new loss function is:
$$Loss=\sum_{p=0}^l{(d_p-h(\mathbf{x}_p))^2}+\lambda||\mathbf{w}||^2$$

The $||\mathbf{w}||^2$ is like the sum over all $w$ $\left(\sum_i w_i^2\right)$.

$\lambda$ is a hyperparameter, so it's search with the hyperparameters search.
If it's applied to a linear model is like the ridge regression.

The general objective is the minimization of the loss function so, with this term, is possible trying to reduce the $||\mathbf{w}||$. 

## Weight dacay
The penalty term $\lambda$ will contribute, as optimizer, in the weights update adding a factor that reduce the weights at each update step.
We can simply add to the $\mathbf{w}_{new}$ the old $\mathbf{w}$ weighted with $\lambda$:
$$\mathbf{w}_{new}=\mathbf{w}+\eta\Delta\mathbf{w}+2\lambda\mathbf{w}$$


## Alternatives 
Besides the L2 it's possible to use other methods for the regularization such as the early stopping.
This consist in stopping the training when are satisfy some criteria in the validation set, for example if the error has few changes.

# Hyperparameters search
We need to do a grid search on 
- $\eta$ : the learning rate
- $\lambda$: the regularization parameter
- $\alpha$: The momentum parameter