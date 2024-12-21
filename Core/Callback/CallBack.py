from Core import FeedForwardModel


class CallBack:

    def Call(self,model:FeedForwardModel):
        pass

    def Reset(self):
        pass

    def __call__(self, model:FeedForwardModel):
        return self.Call(model)