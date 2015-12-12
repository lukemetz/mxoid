import mxnet as mx
import logging
from mxnet.model import _create_kvstore
logger = logging.getLogger(__name__)

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

        self.batch_extentions = []
        self.epoch_extentions = []

        self.log = {}
        self.status = {}
        self.status['iterations'] = 0

        def f(_):
            self.status['iterations'] += 1
        self.batch_extentions.append(f)
        def f(_):
            self.log[self.status['iterations']] = {}
            self.current_log = self.log[self.status['iterations']]
        self.batch_extentions.append(f)

        self.metric = LoopAccumulator(self.sym)

    def add_extention(self, ext):
        ext.loop = self
        self.batch_extentions.append(ext._every_batch)
        self.epoch_extentions.append(ext._every_epoch)

    def run(self):
        data = self.model._init_iter(self.data y, is_train=True)

        arg_names, param_names, aux_names = \
                self.model._init_params(dict(data.provide_data+data.provide_label))

        # create kvstore
        (kvstore, update_on_kvstore) = _create_kvstore(
            kvstore, len(self.ctx), self.arg_params)

        # init optmizer
        if isinstance(self.optimizer, str):
            batch_size = data.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            optimizer = opt.create(self.optimizer,
                                   rescale_grad=(1.0/batch_size),
                                   **(self.kwargs))
        elif isinstance(self.optimizer, opt.Optimizer):
            optimizer = self.optimizer

        self.executor_manager = ExecutorManager(symbol=symbol,
                                           ctx=ctx,
                                           train_data=self.data,
                                           param_names=param_names,
                                           arg_names=arg_names,
                                           aux_names=aux_names,
                                           work_load_list=work_load_list,
                                           logger=logger)
        self.executor_manager.install_monitor(monitor)

        self.executor_manager.set_params(arg_params, aux_params)

        self.updater = UpdateManager(kvstore=kvstore,
                                update_on_kvstore=update_on_kvstore,
                                optimizer=optimizer,
                                param_arrays=executor_manager.param_arrays,
                                arg_params=arg_params,
                                param_names=executor_manager.param_names,
                                ctx=ctx)

        # Now start training
        while True:
            # Training phase
            self.metric.reset()
            nbatch = 0
            # Iterate over training data.
            self.data.reset()
            while True:
                do_reset = True
                for data_batch in self.data:

                    self.executor_manager.load_data_batch(data_batch)

                    self.executor_manager.forward(is_train=True)
                    self.executor_manager.backward()

                    self.updater.do_update(self.executor_manager.param_arrays,
                                           self.executor_manager.grad_arrays)

                    # evaluate at end, so out_cpu_array can lazy copy
                    self.metric.update(data_batch.label, executor_manager.cpu_output_arrays)

                    nbatch += 1

                    # this epoch is done possibly earlier
                    if epoch_size is not None and nbatch >= epoch_size:
                        do_reset = False
                        break

class StubLoop(Loop):
    def __init__(self, sym, data, optimizer, kv=None, devs=[mx.cpu()]):
        self.kv = kv
        self.data = data
        self.optimizer = optimizer
        self.sym = sym
        self.batch_extentions = []
        self.epoch_extentions = []

        self.metric = LoopAccumulator(self.sym)

        self.log = {}
        self.status = {}
        self.status['iterations'] = 0

        def f(_):
            self.status['iterations'] += 1
        self.batch_extentions.append(f)
        def f(_):
            self.log[self.status['iterations']] = {}
            self.current_log = self.log[self.status['iterations']]
        self.batch_extentions.append(f)

    def run(self):
        param = None
        for i in range(100):
            self.metric.num_inst += 1
            for e in self.batch_extentions:
                e(param)
            if i % 10 == 0:
                for e in self.epoch_extentions:
                    e(param)
