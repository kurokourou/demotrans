"""Data generators for  Vietnamese Sentence Refine problem usign CoreZero Boost"""
# Based on TranslateEnviIwslt32k problem
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/translate_envi.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

EOS = text_encoder.EOS_ID

# 40k sentence pairs
# Core sentence extracted from EVBNews 2.0, a part of EVBCorpus
# https://github.com/qhungngo/EVBCorpus # Done
_OUT_BIAS_COREBOOSTER_TRAIN_DATASETS = [[ 
    "https://github.com/kurokourou/demotrans/raw/master/data/corebooster/out_ci_bias/corebooster_train.tar.gz",  
    ("EVBNews_2_0_corebooster.input", "EVBNews_2_0_corebooster.output")
]]

# 5105 sentence pairs used for test
_OUT_BIAS_COREBOOSTER_TEST_DATASETS = [[ 
    "https://github.com/kurokourou/demotrans/raw/master/data/corebooster/out_ci_bias/corebooster_test.tar.gz",  
    ("corebooster_test.input", "corebooster_test.output")
]]

  # 40k sentence pairs # Fixing
  # English core sentence extracted from EVBNews 2.0, a part of EVBCorpus
  # https://github.com/qhungngo/EVBCorpus
  # Core Translate by Core2core_base
# _OUT_BALANCE_COREBOOSTER_TRAIN_DATASETS = [[ # Fix
#     "https://github.com/kurokourou/demotrans/raw/master/data/corebooster/corebooster_train.tar.gz",  
#     ("EVBNews_2_0_corebooster.input", "EVBNews_2_0_corebooster.output")
# ]]

# # 5105 sentence pairs used for test
# _OUT_BALANCE_COREBOOSTER_TEST_DATASETS = [[ # Fix
#     "https://github.com/kurokourou/demotrans/raw/master/data/corebooster/corebooster_test.tar.gz",  
#     ("corebooster_test.input", "corebooster_test.output")
# ]]

# Translate IWLST32 with core sentence intergrated input # Done
_IN_COREBOOSTER_TRAIN_DATASETS = [[ 
    "https://github.com/kurokourou/demotrans/raw/master/data/corebooster/in_ci/train_core.tar.gz",  
    ("train_core.en", "train.vi")
]]

# 5105 sentence pairs used for test
_IN_COREBOOSTER_TEST_DATASETS = [[ 
    "https://github.com/kurokourou/demotrans/raw/master/data/corebooster/in_ci/tst2013_core.tar.gz",  
    ("tst2013_core.en", "tst2013.vi")
]]

@registry.register_problem
class CoreBoostingOutBias(translate.TranslateProblem):
  """ Vietnamese Sentence Sharpener"""
  
  @property
  def approx_vocab_size(self):
    return 2**15  

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _OUT_BIAS_COREBOOSTER_TRAIN_DATASETS if train else _OUT_BIAS_COREBOOSTER_TEST_DATASETS

#Fixing
@registry.register_problem
class CoreBoostingOutBalance(translate.TranslateProblem):
  """ Vietnamese Sentence Sharpener"""
  
  @property
  def approx_vocab_size(self):
    return 2**15  

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _OUT_BALANCE_COREBOOSTER_TRAIN_DATASETS if train else _OUT_BALANCE_COREBOOSTER_TEST_DATASETS

#Fixing
@registry.register_problem
class CoreBoostingIn(translate.TranslateProblem):
  """ Vietnamese Sentence Sharpener"""
  
  @property
  def approx_vocab_size(self):
    return 2**15  

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _IN_COREBOOSTER_TRAIN_DATASETS if train else _IN_COREBOOSTER_TEST_DATASETS    