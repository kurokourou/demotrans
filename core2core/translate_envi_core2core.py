"""Data generators for En-Vi Core Sentence translation."""
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

# 45k sentence pairs
# Core sentence extracted from EVBNews 2.0, a part of EVBCorpus
# https://github.com/qhungngo/EVBCorpus
_ENVI_CORE_TRAIN_DATASETS = [[
    "https://github.com/kurokourou/demotrans/raw/master/data/core2core/core_v1.tar.gz",  
    ("core_v1.en", "core_v1.vi")
]]

# 4425 sentence pairs used for test
# Note: There are 111 sentences duplicated from train set
_ENVI_CORE_TEST_DATASETS = [[
    "https://github.com/kurokourou/demotrans/raw/master/data/core2core/core_v1_test.tar.gz",  
    ("core_v1_test.en", "core_v1_test.vi")
]]

@registry.register_problem
class TranslateEnviCore2core(translate.TranslateProblem):
  """Translate the core_0 of English sentence to Vietnamese"""
  
  @property
  def approx_vocab_size(self):
    return 2**15  

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENVI_CORE_TRAIN_DATASETS if train else _ENVI_CORE_TEST_DATASETS