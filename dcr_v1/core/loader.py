# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import numpy as np
import mxnet as mx
from rcnn import get_rcnn_batch, get_rcnn_test_batch

from mxnet.io import DataDesc, DataBatch
import threading

class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, boxes_all, config, batch_size=1, shuffle=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.boxes_all = boxes_all
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data']
        # self.label_name = ['softmax_label']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = [] # mx.nd.array([])

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]
        # return [
        #     mx.io.DataDesc(k, tuple(list(v.shape)), v.dtype)
        #     for k, v in zip(self.data_name, self.data)
        # ]

    @property
    def provide_label(self):
        return None
        # return [
        #     mx.io.DataDesc(k, tuple(list(v.shape)), v.dtype)
        #     for k, v in zip(self.label_name, self.label)
        # ]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label= self.label, #[], # self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label) #[None]) # self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur > self.size:
            return self.cur - self.size
        # if self.cur + self.batch_size > self.size:
        #     return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        pad = self.batch_size - (cur_to - cur_from)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        boxes_all = [self.boxes_all[self.index[i]] for i in range(cur_from, cur_to)]
        data = get_rcnn_test_batch(roidb, boxes_all, self.cfg, pad)
        self.data = [mx.nd.array(data[name]) for name in self.data_name]


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param config: configuration dictionary
        :param batch_size: total number of images
        :param shuffle: bool
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.cfg = config
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data']
        self.label_name = ['softmax_label']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        data, label = get_rcnn_batch(roidb, self.cfg)

        self.data = [mx.nd.array(data[name]) for name in self.data_name]
        self.label = [mx.nd.array(label[name]) for name in self.label_name]


class PreFetchROIIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param config: configuration dictionary
        :param batch_size: total number of images
        :param shuffle: bool
        :return: ROIIter
        """
        super(PreFetchROIIter, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.cfg = config
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data']
        self.label_name = ['softmax_label']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [
            mx.io.DataDesc(k, tuple(list(v.shape)), v.dtype)
            for k, v in zip(self.data_name, self.data)
        ]

    @property
    def provide_label(self):
        return [
            mx.io.DataDesc(k, tuple(list(v.shape)), v.dtype)
            for k, v in zip(self.label_name, self.label)
        ]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        data, label = get_rcnn_batch(roidb, self.cfg)

        self.data = [mx.nd.array(data[name]) for name in self.data_name]
        self.label = [mx.nd.array(label[name]) for name in self.label_name]


class PrefetchingIter(mx.io.DataIter):
    """Base class for prefetching iterators. Takes one or more DataIters (
    or any class with "reset" and "next" methods) and combine them with
    prefetching. For example:

    Parameters
    ----------
    iters : DataIter or list of DataIter
        one or more DataIters (or any class with "reset" and "next" methods)
    rename_data : None or list of dict
        i-th element is a renaming map for i-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data
    rename_label : None or list of dict
        Similar to rename_data

    Examples
    --------
    iter = PrefetchingIter([NDArrayIter({'data': X1}), NDArrayIter({'data': X2})],
                           rename_data=[{'data': 'data1'}, {'data': 'data2'}])
    """
    def __init__(self, iters, rename_data=None, rename_label=None):
        super(PrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter ==1, "Our prefetching iter only support 1 DataIter"
        self.iters = iters
        self.rename_data = rename_data
        self.rename_label = rename_label
        self.batch_size = self.iters[0].batch_size
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for _ in range(self.n_iter)]
        self.next_batch = [None for _ in range(self.n_iter)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()
        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        if self.rename_data is None:
            return sum([i.provide_data for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_data
            ] for r, i in zip(self.rename_data, self.iters)], [])

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        if self.rename_label is None:
            return sum([i.provide_label for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_label
            ] for r, i in zip(self.rename_label, self.iters)], [])

    def reset(self):
        for e in self.data_ready:
            e.wait()
        for i in self.iters:
            i.reset()
        for e in self.data_ready:
            e.clear()
        for e in self.data_taken:
            e.set()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            return False
        else:
            self.current_batch = self.next_batch[0]
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad