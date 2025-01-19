from Core import FeedForwardModel
from Core.Layer import Layer
from Core.LossFunction import *


class Optimizer:
    """
    Base class for optimization algorithms in neural networks. This class provides the
    foundation for implementing various optimization techniques like gradient descent,
    momentum, and regularization.

    Attributes:
        loss_function (LossFunction): The loss function used to compute the loss and its derivative.
        batchSize (int): The size of the mini-batch for gradient computation.
        eta (float): The learning rate for the optimization process.
        lambda_ (float | None): The regularization factor (L2 regularization). Default is None.
        alpha (float | None): The momentum factor for gradient updates. Default is None.
        decay_rate (float): The learning rate decay factor. Default is 0.0.
        iteration (int): The current iteration in the optimization process.
        deltas (np.ndarray): The list of deltas for each layer, used during backpropagation.
        updates (list): The list of weight updates for each layer.
        momentum (bool): A flag indicating whether momentum is used in the optimization process.
        regularization (bool): A flag indicating whether regularization is applied during optimization.
    """

    batchSize :int

    deltas: np.ndarray[float]
    updates: list[np.ndarray]
    momentum: bool
    regularization: bool

    def __init__(self, loss_function: LossFunction, batchSize:int,eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float = 0.0):
        """
        Base class for optimization algorithms.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        self.loss_function = loss_function
        self.batchSize = batchSize
        self.eta = eta
        self.initial_eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.iteration = 0

        self.deltas = []
        self.updates = []

        if self.alpha is None:
            self.momentum = False
        else:
            self.momentum = True

        if self.lambda_ is None:
            self.regularization = False
        else:
            self.regularization = True


    def StartOptimize(self, model: FeedForwardModel, targets: np.ndarray):
        """
        Starts the optimization process by iterating through all layers in reverse order
        and applying the optimization strategy.

        :param model: The feedforward model containing the layers.
        :param targets: The target values for the optimization.
        """
        for layer in reversed(model.Layers):
            self.Optimize(layer, targets)

        self.iteration += 1
        if self.decay_rate is not None:
            self.eta = self.initial_eta * np.exp(-self.decay_rate * self.iteration)


    def Optimize(self, layer: Layer, target: np.ndarray):
        """
        Central optimization algorithm of the weights for the given layer during backpropagation.

        :param layer: The layer for which to compute the gradients.
        :param target: The target values for the given inputs.
        """

        # Calculate delta
        if layer.LastLayer is None: # Input layer
            self.UpdateWeights(layer)
            return
        else:
            self.deltas = self.CalculateDelta(layer, target)

        # Calculate gradient
        layer_grad = self.eta * self.CalculateGradient(layer)

        # Calculate and apply the momentum
        if self.momentum is True and layer.LastLayer.Gradient is not None:
            layer_grad = self.ApplyMomentum(layer, layer_grad)
            layer.LastLayer.Gradient = layer_grad

        # Apply regularization
        layer_grad = self.ApplyRegularization(layer, layer_grad)

        # Add updates to updates' list
        self.updates.append(layer_grad)

    def CalculateDelta(self, layer: Layer, target: np.ndarray):
        """
        Computes the delta for a layer during backpropagation.

        :param layer: The current layer for which to compute the delta.
        :param target: The target values for the given inputs.
        :return: A numpy array containing the deltas for each unit in the layer.
        """
        deltas = []

        #Output Layer
        if layer.NextLayer is None:
            loss_name = self.loss_function.Name
            activation_name = layer.ActivationFunction.Name

            # Check if the condition for direct subtraction is met
            use_direct_subtraction = (
                    loss_name in {"BinaryCrossEntropyLoss", "CategoricalCrossEntropyLoss"}
                    and activation_name == "SoftARGMax"
            )

            # Compute deltas using vectorized operations
            if use_direct_subtraction: # Special case for SoftMax and CrossEntropy
                deltas = layer.LayerOutput - target
            else:
                loss_derivative = self.loss_function.CalculateDeriv(layer.LayerOutput, target)
                activation_derivative = layer.ActivationFunction.CalculateDerivative(layer.LayerNets)
                deltas = loss_derivative * activation_derivative

        #Hidden Layer
        else:
            prev_delta = self.deltas

            weights = self.PreProcessWeights(layer)
            # Transpose the weight matrix to focus on individual unit outputs

            all_weight_T = weights.T
            # If bias is used, exclude the last row
            if layer.UseBias:
                all_weight_T = all_weight_T[:-1]

            # Compute the weighted sum of deltas from the previous layer
            delta_sum = all_weight_T @ prev_delta.T

            activation_derivative = layer.ActivationFunction.CalculateDerivative(layer.LayerNets.T)
            deltas = (delta_sum * activation_derivative).T



        return deltas

    def CalculateGradient(self, layer: Layer):
        """
        Computes the gradient of the weights for a layer using the deltas.

        :param layer: The layer for which to compute the gradients.
        :return: The computed gradient of the weights for the layer.
        """

        # Compute gradients using broadcasting
        gradients = layer.LayerInput[:, :, None] * self.deltas[:, None, :]  # Shape: (batch_size, input_dim, output_dim)


        # Average gradients across the batch
        layer_grad = np.mean(gradients, axis=0)

        # equivalent to
        # for prev_o_one_pattern, deltas_one_pattern in zip(layer.LayerInput, self.deltas):
        #     b = np.expand_dims(prev_o_one_pattern, axis=1)
        #     out_delta_p = np.expand_dims(deltas_one_pattern, axis=0)
        #     grad_from_one_unit = b @ out_delta_p
        #     layer_grad.append(grad_from_one_unit.T)
        # layer_grad = np.mean(np.array(layer_grad), axis=0)

        return layer_grad.T

    def UpdateWeights(self, layer: Layer):
        """
        Function to update the layer's weights by applying the computed gradients.

        :param layer: The layer to update.
        """
        index = 1
        max_norm = 1.0  # Define the maximum norm for gradient clipping
        while layer.NextLayer is not None:
            # Clip the gradients before updating weights
            gradients = self.updates[-index]
            norm = np.linalg.norm(gradients)
            if norm > max_norm:
                gradients = gradients * (max_norm / norm)

            # Update the weights in the layer
            layer.WeightToNextLayer = layer.WeightToNextLayer - gradients
            layer = layer.NextLayer
            index += 1
        self.updates = []

    def ApplyRegularization(self, layer: Layer, layer_grad: np.ndarray):
        """
        Applies L2 regularization to the gradients of the weights.

        :param layer: The current layer for which regularization is applied.
        :param layer_grad: The gradient of the weights for the layer.
        :return: The updated gradient after applying regularization.
        """
        if not self.regularization:
            return layer_grad

        # Extract weights and biases separately if the layer uses biases
        if layer.LastLayer.UseBias:
            weights = layer.LastLayer.WeightToNextLayer[:, :-1]  # Exclude bias from weights
            bias_grad = layer_grad[:, -1]  # Bias gradient
            weight_grad = layer_grad[:, :-1]  # Weight gradient
            weight_grad_with_reg = weight_grad + (self.lambda_ * weights)
            # Combine updated weights with original bias gradient
            layer_update = np.hstack((weight_grad_with_reg, bias_grad[:, np.newaxis]))
        else:
            weights = layer.LastLayer.WeightToNextLayer  # All weights
            weight_grad = layer_grad
            weight_grad_with_reg = weight_grad + ( self.lambda_ * weights)
            layer_update = weight_grad_with_reg


        # Apply L2 regularization to weight gradients
        return layer_update



    def PreProcessWeights(self, layer: Layer):
        """
        Preprocesses the weights for the layer before applying momentum.

        :param layer: The current layer for which to preprocess the weights.
        :return: The preprocessed weights.
        """
        return layer.WeightToNextLayer

    def ApplyMomentum (self, layer: Layer, layer_grad: np.ndarray):
        """
        Applies momentum to the gradient during optimization.

        :param layer: The current layer for which to apply momentum.
        :param layer_grad: The computed gradient of the weights.
        :return: The updated gradient with momentum applied.
        """
        if layer.LastLayer.Gradient is None:
            return None

        if layer.LastLayer.Gradient.shape != layer_grad.shape:
            # Pad gradients to match shape, this is used only in the last mini-batch that usually have fewer data
            pad_size = layer.LastLayer.Gradient.shape[0] - layer_grad.shape[0]
            layer_grad = np.pad(layer_grad, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
        velocity = self.alpha * layer.LastLayer.Gradient
        layer_grad = layer_grad + velocity

        return layer_grad