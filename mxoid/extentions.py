import time
import json

class Extention(object):
    def __init__(self, every_epoch=False, every_n_batches=None):
        self.every_epoch = every_epoch
        self.every_n_batches = every_n_batches
        self.loop = None

    def _every_batch(self, params):
        if self.every_n_batches and self.loop.status['iterations'] % self.every_n_batches == 0:
            self.do(self.loop)

    def _every_epoch(self, params, *args, **kwargs):
        if self.every_epoch:
            self.do(self.loop)

    def do(self, loop):
        raise NotImplemented()

class ExamplesPerSecond(Extention):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.init = False
        self.tic = 0
        self.last_count = 0
        super(ExamplesPerSecond, self).__init__(**kwargs)

    def do(self, loop):
        """Callback to Show speed."""
        count = loop.status['iterations']
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            speed = self.every_n_batches * self.batch_size / (time.time() - self.tic)
            loop.current_log['examples_per_second'] = speed
            self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

class TrainMonitor(Extention):
    def __init__(self, prefix, **kwargs):
        self.prefix = prefix
        super(TrainMonitor, self).__init__(**kwargs)
    def do(self, loop):
        names = loop.metric.sym.list_outputs()
        for name, val in zip(names, loop.metric.buff):
            loop.current_log["%s_%s"%(self.prefix, name)] = val / loop.metric.num_inst

        loop.metric.reset()

class EvalMonitor(Extention):
    def __init__(self, symbol, prefix, **kwargs):
        super(EvalMonitor, self).__init__(**kwargs)
    def do(self, loop):
        pass

class Printing(Extention):
    def do(self, loop):
        print "=============================="
        print "== Status =="
        for name,val in loop.status.items():
            print "\t%s: %s"%(name, str(val))
        print "== Log =="
        for name, val in loop.current_log.items():
            print "\t%s: %s"%(name, str(val))

class NDJsonLogger(Extention):
    def __init__(self, filepath, **kwargs):
        self.file_ = open(filepath, "w")
        self.keys_written = set()
        super(NDJsonLogger, self).__init__(**kwargs)

    def do(self, loop):
        to_write = set(loop.log.keys()) - self.keys_written
        for r in sorted(list(to_write)):
            self.file_.write(json.dumps(loop.log[r])+"\n")
        self.file_.flush()
        self.keys_written = self.keys_written.union(to_write)


