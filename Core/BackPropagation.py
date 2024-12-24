from Core import FeedForwardModel
from Core.ActivationFunction import *
from Core.Layer import Layer
from Core.LossFunction import *


class Optimizer:
    deltas: list[float]
    updates: list[np.ndarray]
    momentum: bool
    regularization: bool

    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None):
        """
        Base class for optimization algorithms.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        self.loss_function = loss_function
        self.eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha

        self.deltas = []
        self.updates = []

    def start_optimize(self, model: FeedForwardModel, target: np.ndarray):
        for layer in reversed(model.Layers):
            self.optimize(layer, target)


    def optimize(self, layer: Layer, target: np.ndarray):
        """
        Abstract method to compute gradients for a given layer.
        """
        raise NotImplementedError("Must override Calculate method")

    def update_weights(self, layer: Layer):
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


class BackPropagation(Optimizer):
    def __init__(self, loss_function: LossFunction, eta: float | None, lambda_: float | None = None,
                 alpha: float | None = None):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function, eta, lambda_, alpha)
        if self.alpha is None:
            self.momentum = False
        else:
            self.momentum = True

        if self.lambda_ is None:
            self.regularization = False
        else:
            self.regularization = True

    def optimize(self, layer: Layer, target: np.ndarray):
        """
        Computes the gradients of the weights for the given layer during backpropagation.

        :param layer: The layer for which to compute the gradients.
        :param target: The target values for the given inputs.
        """

        act_fun = layer.ActivationFunction

        deltas =[]

        if not layer.TrainMode:
            self.updates.append(np.zeros_like(layer.LastLayer.WeightToNextLayer))

        # Calculate delta
        if layer.LastLayer is None: # Input layer
            # Start the weight updates
            self.update_weights(layer)
            return
        elif layer.NextLayer is None:
            for out, y, net in zip(layer.LayerOutput, target, layer.LayerNets):
                out_delta_p = self.loss_function.CalculateDerivLoss(out, y) *act_fun.CalculateDerivative(net)
                deltas.append(out_delta_p)
            deltas = np.array(deltas)

        else: # Hidden layer
            prev_delta = self.deltas

            # transpose weight metric to use a single unit's output weights
            all_weight_T = layer.WeightToNextLayer.T

            if layer.UseBias is True:
                # get all weight exept the bias
                unit_weight_T = all_weight_T[:-1]
                # get only the bias weight
                bias_weight_T = all_weight_T[-1]
            else:
                unit_weight_T= all_weight_T
                bias_weight_T= None



            for weightFrom1Neuron_toAll, net_one_unit in zip(unit_weight_T, layer.LayerNets.T):
                # compute the internal sum for a single unit
                delta_sum = (prev_delta @ weightFrom1Neuron_toAll)
                delta = delta_sum * act_fun.CalculateDerivative(net_one_unit)
                deltas.append(delta)

            if layer.UseBias:
                #Calcualte Bias delta
                delta_sum = (prev_delta @ bias_weight_T)
                deltas.append(delta_sum)

            deltas = np.array(deltas).T


        # Calculate Gradient
        if layer.UseBias is True:
            deltas = deltas[:,:-1]
        # Save Delta for the next layer
        self.deltas = deltas

        layer_grad = []
        # this is more efficient, but the comment below explains the logic of the operation
        # layerGrad = self.eta * np.einsum('ij,ik->ikj', layer.LayerInput, deltas)

        for prev_o_one_pattern, deltas_one_pattern in zip(layer.LayerInput, deltas):
            b = np.expand_dims(prev_o_one_pattern, axis=1)
            out_delta_p = np.expand_dims(deltas_one_pattern, axis=0)
            grad_from_one_unit = self.eta * (b @ out_delta_p)
            layer_grad.append(grad_from_one_unit.T)
        layerGrad = np.mean(np.array(layer_grad), axis=0)


        # Calculte and applay the momentum
        if self.momentum is True:
            if layer.LastLayer.Gradient is not None:
                if layer.LastLayer.Gradient.shape != layerGrad.shape:
                    # Pad gradients to match shape (this is used expecialy in the last mini-batch that usualy have less data)
                    pad_size = layer.LastLayer.Gradient.shape[0] - layerGrad.shape[0]
                    layerGrad = np.pad(layerGrad, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
                mom = self.alpha * layer.LastLayer.Gradient
                layerGrad = layerGrad + mom
            layer.LastLayer.Gradient = layerGrad


    # Optimize the weights
        if self.regularization is True:
            layerUpdate = layerGrad - ( 2 * self.lambda_ * layer.LastLayer.WeightToNextLayer)
        else:
            layerUpdate = layerGrad

        self.updates.append(layerUpdate)


"""
class BackPropNesterovMomentum(Optimizer):
    def __init__(self, loss_function: LossFunction, eta: float, alpha: float, lambda_: float | None = None):

        Initializes the NesterovMomentum optimizer.

        :param loss_function: Loss function instance.
        :param eta: Learning rate.
        :param alpha: Momentum coefficient.
        :param lambda_: L2 regularization parameter (optional).

        super().__init__(loss_function, eta, lambda_, alpha)
        self.velocity = None  # Initialize the velocity to None

    def optimize(self, layer: Layer, target: np.ndarray):

        Optimizes weights of a layer using Nesterov momentum.

        :param layer: The layer to optimize.
        :param target: Target values for the given inputs.

                # Input layer
        if layer.LastLayer is None:
            raise Exception("Iteration on the wrong layer")

        # Initialize velocity if it's the first iteration
        if self.velocity is None:
            self.velocity = np.zeros_like(layer.LastLayer.WeightToNextLayer)

        # Lookahead step
        lookahead_weights = layer.LastLayer.WeightToNextLayer + (self.alpha * self.velocity)

        # Compute delta (error term) at lookahead position
        if layer.NextLayer is None:  # Output layer
            delta = (
                self.loss_function.CalculateDerivLoss(layer.LayerOutput, target)
                * layer.ActivationFunction.CalculateDerivative(layer.LayerNets)
            )
        else:  # Hidden layer
            delta = np.dot(self.deltas, layer.LastLayer.WeightToNextLayer.T) * layer.ActivationFunction.CalculateDerivative(layer.LayerNets)

        self.deltas = delta

        # Calculate gradient at the lookahead position
        gradient = np.dot(self.deltas.T, layer.LayerInput)

        # Apply L2 regularization if enabled
        if self.lambda_ is not None:
            gradient += self.lambda_ * lookahead_weights

        # Update velocity and weights
        self.velocity = self.alpha * self.velocity - self.eta * gradient
        layer.LastLayer.WeightToNextLayer += self.velocity
"""