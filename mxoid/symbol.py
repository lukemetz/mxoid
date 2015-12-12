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

