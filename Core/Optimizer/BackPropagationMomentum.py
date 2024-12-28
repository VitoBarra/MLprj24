from Core.Layer import Layer
from Core.LossFunction import *
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationMomentum(Optimizer):

    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float | None = 0.0):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function, eta, lambda_, alpha, decay_rate)
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


        deltas =[]

        # Calculate delta
        if layer.LastLayer is None: # Input layer
            self.update_weights(layer)
            return
        elif layer.NextLayer is None:
            for out, y, net in zip(layer.LayerOutput, target, layer.LayerNets):
                out_delta_p = self.loss_function.CalculateDerivLoss(out, y) * layer.ActivationFunction.CalculateDerivative(net)
                deltas.append(out_delta_p)
            deltas = np.array(deltas)

        else: # Hidden layer
            prev_delta = self.deltas



            # transpose weight metric to use a single unit's output weights
            all_weight_T = layer.WeightToNextLayer.T

            if layer.UseBias is True:
                # get all weight except the bias
                unit_weight_T = all_weight_T[:-1]
                # get only the bias weight
                bias_weight_T = all_weight_T[-1]
            else:
                unit_weight_T= all_weight_T
                bias_weight_T= None



            for weightFrom1Neuron_toAll, net_one_unit in zip(unit_weight_T, layer.LayerNets.T):
                # compute the internal sum for a single unit
                delta_sum = (prev_delta @ weightFrom1Neuron_toAll)
                delta = delta_sum * layer.ActivationFunction.CalculateDerivative(net_one_unit)
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


        # Calculate and apply the momentum
        if layer.LastLayer.Velocity is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1-self.alpha) * layerGrad)
        else:
            velocity = layer.LastLayer.Velocity * self.alpha + ((1 - self.alpha) * layerGrad)
        layer.LastLayer.Velocity = velocity

        layerGrad = velocity + layerGrad

        # Optimize the weights
        if self.regularization is True:
            layerUpdate = layerGrad + ( 2 * self.lambda_ * layer.LastLayer.WeightToNextLayer)
        else:
            layerUpdate = layerGrad

        self.updates.append(layerUpdate)


