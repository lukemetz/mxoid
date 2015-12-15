
class LoopAccumulator(object):
    def __init__(self, sym, **kwargs):
        self.sym = sym
        self.buff = [0 for _ in self.sym.list_outputs()]
        self.num_inst = 0

    def update(self, label, pred):
        for i, p in enumerate(pred):
            self.buff[i] += p.asnumpy().sum()
        self.num_inst += pred[0].shape[0]

    def reset(self):
        self.buff = [0 for _ in self.sym.list_outputs()]
        self.num_inst = 0
