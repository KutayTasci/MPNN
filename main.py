import experiments.qm9_exp as qm9_exp
import experiments.flickr_exp as flickr_exp
import torch.profiler

#flickr_exp.FlickrExperiment(benchmark=True, model_type='sum')
qm9_exp.QM9Experiment(benchmark=True, model_type='cat', batch_size=32000)
qm9_exp.QM9Experiment(benchmark=True, model_type='sum', batch_size=32000)



