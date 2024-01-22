class SapiensKNN:
    def __init__(self, k=1, normalization=False, regression=False):
        from .core import SapiensKNN
        self.__sapiensknn = SapiensKNN(k=k, normalization=normalization, regression=regression)
    def fit(self, inputs=[], outputs=[]): self.__sapiensknn.fit(inputs=inputs, outputs=outputs)
    def saveModel(self, path=''): self.__sapiensknn.saveModel(path=path)
    def loadModel(self, path=''): self.__sapiensknn.loadModel(path=path)
    def transferLearning(self, transmitter_path='', receiver_path='', rescue_path=''): self.__sapiensknn.transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, rescue_path=rescue_path)
    def predict(self, inputs=[]): return self.__sapiensknn.predict(inputs=inputs)
    def test(self, inputs=[], outputs=[]): return self.__sapiensknn.test(inputs=inputs, outputs=outputs)
