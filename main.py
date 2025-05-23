
import experiments.test_experiment as test_exp
import logging
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

test_exp.test_egnn(benchmark=True, model_type='cat', batch_size=1)
test_exp.test_egnn(benchmark=True, model_type='sum', batch_size=1)

test_exp.test_egnn(benchmark=True, model_type='sum', batch_size=1)
test_exp.test_egnn(benchmark=True, model_type='cat', batch_size=1)


