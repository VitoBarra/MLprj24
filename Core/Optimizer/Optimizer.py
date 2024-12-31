from Core import FeedForwardModel
from Core.Layer import Layer
from Core.LossFunction import *


class Optimizer:
    deltas: np.ndarray[float]
    updates: list[np.ndarray]
    momentum: bool
    regularization: bool

    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float = 0.0):
        """
        Base class for optimization algorithms.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        self.loss_function = loss_function
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


    def StartOptimize(self, model: FeedForwardModel, target: np.ndarray):
        for layer in reversed(model.Layers):
            self.optimize(layer, target)

        self.iteration += 1
        if self.decay_rate is not None:
            self.eta = self.initial_eta * np.exp(-self.decay_rate * self.iteration)


    def optimize(self, layer: Layer, target: np.ndarray):
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

        # Calculte and applay the momentum
        if self.momentum is True and layer.LastLayer.Gradient is not None:
            layer_grad = self.ApplyMomentum(layer, layer_grad)
            layer.LastLayer.Gradient = layer_grad

        # Apply regularization
        layer_grad = self.ApplayRegularization(layer, layer_grad)

        # Add updates to updates' list
        self.updates.append(layer_grad)

    def CalculateDelta(self, layer: Layer, target: np.ndarray):
        deltas = []

        #Output Layer
        if layer.NextLayer is None:
            for out, y, net in zip(layer.LayerOutput, target, layer.LayerNets):
                out_delta_p = self.loss_function.CalculateDerivLoss(out,
                                                                    y) * layer.ActivationFunction.CalculateDerivative(
                    net)
                deltas.append(out_delta_p)
            deltas = np.array(deltas)

        #Hidden Layer
        else:
            prev_delta = self.deltas
            weights = self.PreProcessigWeights(layer)

            # transpose weight metric to use a single unit's output weights
            all_weight_T = weights.T

            if layer.UseBias is True:
                # get all weight except the bias
                unit_weight_T = all_weight_T[:-1]
                # get only the bias weight
                bias_weight_T = all_weight_T[-1]
            else:
                unit_weight_T = all_weight_T
                bias_weight_T = None

            for weightFrom1Neuron_toAll, net_one_unit in zip(unit_weight_T, layer.LayerNets.T):
                # compute the internal sum for a single unit
                delta_sum = (prev_delta @ weightFrom1Neuron_toAll)
                delta = delta_sum * layer.ActivationFunction.CalculateDerivative(net_one_unit)
                deltas.append(delta)

            if layer.UseBias:
                # Calcualte Bias delta
                delta_sum = (prev_delta @ bias_weight_T)
                deltas.append(delta_sum)

            deltas = np.array(deltas).T

        if layer.UseBias is True:
            deltas = deltas[:, :-1]

        return deltas

    def CalculateGradient(self, layer: Layer):
        layer_grad = []

        for prev_o_one_pattern, deltas_one_pattern in zip(layer.LayerInput, self.deltas):
            b = np.expand_dims(prev_o_one_pattern, axis=1)
            out_delta_p = np.expand_dims(deltas_one_pattern, axis=0)
            grad_from_one_unit = b @ out_delta_p
            layer_grad.append(grad_from_one_unit.T)
        layer_grad = np.mean(np.array(layer_grad), axis=0)

        return layer_grad

    def UpdateWeights(self, layer: Layer):
        """
        Function to update the layer's weights.

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

    def ApplayRegularization(self, layer: Layer, layer_grad: np.ndarray):
        # Optimize the weights
        if not self.regularization :
            return layer_grad


        if layer.UseBias:
            unit_weights = layer.LastLayer.WeightToNextLayer[:-1]
            layer_grad = layer_grad[:-1]
            layer_grad_bias = layer_grad[-1]
        else:
            unit_weights = layer.LastLayer.WeightToNextLayer

        layer_update = layer_grad + (2 * self.lambda_ * unit_weights)

        if layer.UseBias:
            # Add back the bias gradient
            layer_update = np.vstack((layer_update, layer_grad_bias[np.newaxis, :]))
        else:
            layer_update = layer_grad

        return layer_update

    def PreProcessigWeights(self, layer: Layer):
        return layer.WeightToNextLayer

    def ApplyMomentum (self, layer: Layer, layer_grad: np.ndarray):
        if layer.LastLayer.Gradient is None:
            return None

        if layer.LastLayer.Gradient.shape != layer_grad.shape:
            # Pad gradients to match shape, this is used only in the last mini-batch that usually have fewer data
            pad_size = layer.LastLayer.Gradient.shape[0] - layer_grad.shape[0]
            layer_grad = np.pad(layer_grad, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
        velocity = self.alpha * layer.LastLayer.Gradient
        layer_grad = layer_grad + velocity

        return layer_grad