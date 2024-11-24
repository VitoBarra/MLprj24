class ModelFeedForward:
    def __init__(self):
        self.Layers = []
        self.InputLayer = None
        self.OutputLayer = None
        self.Optimizer = None
        self.Metric = []
        self.MetricResults = []

    def Fit(self, epoch, batchSize):
        batch_generator = MiniBatchGenerator(self.InputLayer.LastOutputs, batchSize)
        for e in range(epoch):
            batch_generator.Reset()
            while True:
                batch = batch_generator.NextBatch()
                if batch is None:
                    break
                inputs, targets = batch
                outputs = self._forward(inputs)
                self.Optimizer.BackPropagation(outputs, targets)
                self._update_weights()
            self._compute_metrics(e)

    def AddLayer(self, newLayer):
        if self.Layers:
            self.Layers[-1].NextLayer = newLayer
            newLayer.LastLayer = self.Layers[-1]
        self.Layers.append(newLayer)
        if not self.InputLayer:
            self.InputLayer = newLayer
        self.OutputLayer = newLayer

    def SaveModel(self, path):
        model_data = {
            "layers": [layer.get_weights() for layer in self.Layers],
        }
        with open(path, 'wb') as f:
            np.save(f, model_data)

    def LoadModel(self, path):
        with open(path, 'rb') as f:
            model_data = np.load(f, allow_pickle=True).item()
        for layer, weights in zip(self.Layers, model_data["layers"]):
            layer.set_weights(weights)

    def SetOptimizer(self, optimizer):
        self.Optimizer = optimizer

    def AddMetric(self, metric):
        self.Metric.append(metric)

    def AddMetrics(self, metrics):
        self.Metric.extend(metrics)

    def _forward(self, inputs):
        for layer in self.Layers:
            inputs = layer.Compute(inputs)
        return inputs

    def _update_weights(self):
        for layer in reversed(self.Layers):
            layer.Update(self.Optimizer)

    def _compute_metrics(self, epoch):
        results = [metric.Error(self.OutputLayer.LastOutputs, self.OutputLayer.targets) for metric in self.Metric]
        self.MetricResults.append(results)
