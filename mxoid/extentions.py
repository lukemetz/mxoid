import time
import json
from accumulator import LoopAccumulator
import os
import shutil
import datetime
from mxnet.model import save_checkpoint
import logging
logger = logging.getLogger(__name__)

class Extention(object):
    def __init__(self, every_epoch=False, every_n_batches=None, before_training=False):
        self.every_epoch = every_epoch
        self.every_n_batches = every_n_batches
        self.before_training = before_training

    def _every_batch(self, loop):
        if self.every_n_batches and loop.status['iterations'] % self.every_n_batches == 0:
            self.do(loop)

    def _every_epoch(self, loop):
        if self.every_epoch:
            self.do(loop)

    def _before_training(self, loop):
        if self.before_training:
            self.do(loop)

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
    def __init__(self, data, symbol, prefix, **kwargs):
        super(EvalMonitor, self).__init__(**kwargs)
        self.metric = LoopAccumulator(symbol)
        self.prefix = prefix
        self.data = data

    def do(self, loop):
        self.metric.reset()
        self.data.reset()
        for data_batch in self.data:
            loop.executor_manager.load_data_batch(data_batch)

            loop.executor_manager.forward(is_train=False)
            # evaluate at end, so out_cpu_array can lazy copy
            self.metric.update(data_batch.label, loop.executor_manager.cpu_output_arrays)

        names = loop.metric.sym.list_outputs()
        for name, val in zip(names, self.metric.buff):
            loop.current_log["%s_%s"%(self.prefix, name)] = val / self.metric.num_inst

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
        self.keys_written = set()
        self.file_ = None
        self.make_file = lambda : open(filepath, "w")
        super(NDJsonLogger, self).__init__(**kwargs)

    def do(self, loop):
        if self.file_ is None:
            self.file_ = self.make_file()
        to_write = set(loop.log.keys()) - self.keys_written
        for r in sorted(list(to_write)):
            self.file_.write(json.dumps(loop.log[r])+"\n")
        self.file_.flush()
        self.keys_written = self.keys_written.union(to_write)

class FuncExtention(Extention):
    def __init__(self, func, **kwargs):
        self.func = func
        super(FuncExtention, self).__init__(**kwargs)

    def do(self, loop):
        self.func(loop)

class DirectoryCreator(Extention):
    def __init__(self, directory, **kwargs):
        kwargs.setdefault("before_training", True)
        self.once = False
        self.directory = directory
        super(DirectoryCreator, self).__init__(**kwargs)

    def do(self, loop):
        assert not self.once
        self.once = True
        directory = self.directory
        if directory[-1] == "/":
            directory = directory[:-1]
        if os.path.exists(directory):
            time_string = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
            move_to = directory + time_string + "_backup"
            shutil.move(directory, move_to)
        os.mkdir(directory)
        exp_name = directory.split("/")[-1]
        loop.status['exp_name'] = exp_name

class SourceSaver(Extention):
    """
    Save the source to a given folder

    Parameters
    ---------
    dest_directory: basestring
        Path to dump the experiment.
    src_directory: basestring
        Path to source to be copied.
    """

    def __init__(self, dest_directory, src_directory, **kwargs):
        self.dest_directory = dest_directory
        self.src_directory = src_directory
        self.once = False
        kwargs.setdefault("before_training", True)

        super(SourceSaver, self).__init__(**kwargs)

    def write_src(self):
        src_path = os.path.join(self.dest_directory, 'src')
        def ignore(path, names):
            # TODO actually manage paths correctly
            if path == self.dest_directory or\
               path == './' + self.dest_directory:

                return names
            else:
                return []

        shutil.copytree(self.src_directory, src_path, ignore=ignore)

    def do(self, loop):
        assert self.once == False
        self.once = True
        self.write_src()

class SaveCheckpoint(Extention):
    def __init__(self, prefix,  **kwargs):
        self.prefix = prefix
        kwargs.setdefault("every_epoch", True)
        super(SaveCheckpoint, self).__init__(**kwargs)
    def do(self, loop):
        logging.info("Saving model to %s @ %d"%(self.prefix, loop.status['epochs']))
        save_checkpoint(self.prefix, loop.status['epochs'], loop.sym, loop.model.arg_params, loop.model.aux_params)
