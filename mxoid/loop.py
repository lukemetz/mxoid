import mxnet as mx
from mxnet.executor import ExecutorManager
from mxnet.optimizer import UpdateManager
import logging
from accumulator import LoopAccumulator
from mxnet.model import _create_kvstore
logger = logging.getLogger(__name__)


class Loop(object):
    def __init__(self, sym, data, optimizer, kv=None, devs=[mx.cpu()]):
        # kvstore
        if kv == None:
            kv = mx.kvstore.create("local")
        self.kv = kv
        self.data = data
        self.optimizer = optimizer
        self.sym = sym

        self.model = mx.model.FeedForward(
            ctx                = devs,
            symbol             = sym,
            num_epoch          = 1000,
            optimizer          = optimizer,
            wd                 = 0.00001,
            initializer        = mx.init.Normal(0.02))

        self.ctxs = devs
        self.batch_extentions = []
        self.epoch_extentions = []
        self.before_training_extentions = []

        self.log = {}
        self.status = {}
        self.status['iterations'] = 0
        self.status['epochs'] = 0
        self.status['epoch_iterations'] = 0

        self.metric = LoopAccumulator(self.sym)

    def add_extention(self, ext):
        self.batch_extentions.append(ext._every_batch)
        self.epoch_extentions.append(ext._every_epoch)
        self.before_training_extentions.append(ext._before_training)

    def run(self):
        data = self.model._init_iter(self.data, None, is_train=True)

        arg_names, param_names, aux_names = \
                self.model._init_params(dict(data.provide_data+data.provide_label))

        # create kvstore
        (kvstore, update_on_kvstore) = _create_kvstore(
            self.kv, len(self.ctxs), self.model.arg_params)

        self.executor_manager = ExecutorManager(symbol=self.sym,
                                                ctx=self.ctxs,
                                                train_data=self.data,
                                                param_names=param_names,
                                                arg_names=arg_names,
                                                aux_names=aux_names,
                                                logger=logger)

        self.executor_manager.set_params(self.model.arg_params, self.model.aux_params)

        self.updater = UpdateManager(kvstore=kvstore,
                                update_on_kvstore=update_on_kvstore,
                                optimizer=self.optimizer,
                                param_arrays=self.executor_manager.param_arrays,
                                arg_params=self.model.arg_params,
                                param_names=self.executor_manager.param_names,
                                ctx=self.ctxs)

        for e in self.before_training_extentions:
            e(self)

        # Now start training
        while True:
            self.metric.reset()
            nbatch = 0
            self.data.reset()

            for data_batch in self.data:
                self.executor_manager.load_data_batch(data_batch)

                self.executor_manager.forward(is_train=True)
                self.executor_manager.backward()

                self.updater.do_update(self.executor_manager.param_arrays,
                                       self.executor_manager.grad_arrays)

                # evaluate at end, so out_cpu_array can lazy copy
                self.metric.update(data_batch.label, self.executor_manager.cpu_output_arrays)

                self.status['iterations'] += 1
                self.status['epoch_iterations'] += 1
                self.log[self.status['iterations']] = dict(iterations=self.status['iterations'])
                self.current_log = self.log[self.status['iterations']]

                for e in self.batch_extentions:
                    e(self)
                nbatch += 1
            self.status['epochs'] += 1
            self.status['epoch_iterations'] = 0

            for e in self.epoch_extentions:
                e(self)
