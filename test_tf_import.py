#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.WARNING)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    print('TF_AVAILABLE = True')
    print(f'TensorFlow version: {tf.__version__}')
except ImportError as e:
    print('TF_AVAILABLE = False, Error:', e)
except Exception as e:
    print('TF_AVAILABLE = False, Other error:', e)
