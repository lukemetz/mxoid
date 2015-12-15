import numpy as np
import mxnet as mx


class Misclassification(mx.operator.NumpyOp):
    def __init__(self, top_k):
        self.top_k = top_k
        super(Misclassification, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['label', 'data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[1]
        label_shape = (in_shape[1][0],)
        output_shape = (in_shape[1][0],)
        return [label_shape, data_shape], [output_shape]
    def backward(self, out_grad, in_data, out_data, in_grad):
        pass
    def forward(self, in_data, out_data):
        y = in_data[0].astype("int32")
        y_hat = in_data[1]
        row_offsets = np.arange(0, y_hat.flatten().shape[0], y_hat.shape[1])
        truth_score = y_hat.flatten()[row_offsets + y]
        higher_scoring = y_hat >= truth_score.reshape((-1, 1))
        num_higher = higher_scoring.sum(axis=1) - 1
        out_data[0][:] = num_higher >= self.top_k

class BinaryCrossEntropy(mx.operator.NumpyOp):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['label', 'data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        print in_shape
        data_shape = in_shape[1]
        label_shape = (data_shape[0], )
        out_shape = (data_shape[0], )
        o = [label_shape, data_shape], [out_shape]
        return o
    def forward(self, in_data, out_data):
        y = in_data[0].ravel()
        y_hat = in_data[1].ravel()
        print y.shape, y_hat.shape, out_data[0].shape
        print "FINISHeD?"
        out_data[0][:] = (-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
    def backward(self, out_grad, in_data, out_data, in_grad):
        y = in_data[0].ravel()
        y_hat = in_data[1].ravel()
        print in_grad[0].shape
        #print out_grad[0].shape
        print "backpass"
        #in_grad[0][:] = (y_hat - y)# * out_grad[0]
        in_grad[1][:] = (y_hat - y).reshape(-1,1)# * out_grad[0]
        print in_grad[0]

        print "^^^"
