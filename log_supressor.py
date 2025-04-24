import os
import sys
import logging
import absl.logging

def mute_tensorflow_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('absl').setLevel(logging.FATAL)
    absl.logging.set_verbosity(absl.logging.FATAL)
    absl.logging.set_stderrthreshold('fatal')
    sys.stderr = open(os.devnull, 'w')  # optional: full silence

