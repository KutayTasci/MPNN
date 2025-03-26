import experiments.qm9_exp as qm9_exp
import experiments.flickr_exp as flickr_exp
import experiments.s3dis_exp as s3dis_exp
import experiments.modelnet_exp as modelnet_exp
import torch.profiler

#flickr_exp.FlickrExperiment(benchmark=True, model_type='sum')
#qm9_exp.QM9Experiment(benchmark=True, model_type='cat', batch_size=32000)

modelnet_exp.ModelNetExperiment(benchmark=True, model_type='cat')
modelnet_exp.ModelNetExperiment(benchmark=True, model_type='sum')



