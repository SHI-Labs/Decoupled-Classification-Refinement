# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import mxnet as mx
import time

def test_io(DataIter, epochs, disp_batches, batch_size, logger=None):
    tic = time.time()
    print 'Total epoch: %d' % epochs
    for epoch in range(epochs):
        DataIter.reset()
        for i, batch in enumerate(DataIter):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % disp_batches == 0:
                str = 'Epoch [%d] Batch [%d]\tSpeed: %.2f samples/sec' % (
                    epoch, i+1, disp_batches * batch_size / (time.time() - tic))
                print str
                if logger is not None:
                    logger.info(str)
                tic = time.time()