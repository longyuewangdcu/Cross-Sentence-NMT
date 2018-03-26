# data stream

import sys
sys.path.append('/path/code_lc')
print sys.path
import configurations
import argparse

import logging
import cPickle as pkl
import os
import numpy
from fuel.datasets import TextFile
from fuel.streams import DataStream


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DStream(object):

    def __init__(self, **kwards):

        self.train_src = kwards.pop('train_src')
        self.train_trg = kwards.pop('train_trg')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')

        # added by Longyue
        self.train_src_hist = kwards.pop('train_src_hist')
        self.hist_len = kwards.pop('hist_len')

        self.unk_token = kwards.pop('unk_token')
        self.unk_id = kwards.pop('unk_id')
        self.eos_token = kwards.pop('eos_token')
        self.src_vocab_size = kwards.pop('src_vocab_size')
        self.trg_vocab_size = kwards.pop('trg_vocab_size')
        self.seq_len_src = kwards.pop('seq_len_src')
        self.seq_len_trg = kwards.pop('seq_len_trg')
        self.batch_size = kwards.pop('batch_size')
        self.sort_k_batches = kwards.pop('sort_k_batches')

        # get source and target dicts
        self.src_dict, self.trg_dict = self._get_dict()
        self.eos_id = self.src_dict[self.eos_token]

        # convert senteces to ids and filter length > seq_len_src / seq_len_trg
        # modified by Longyue
        self.source, self.source_hist, self.target = self._get_sentence_pairs()

        # sorted k batches
        if self.sort_k_batches > 1:
            self.source, self.source_hist, self.target = self._sort_by_k_batches(self.source, self.source_hist, self.target)

        num_sents = len(self.source)
        assert num_sents == len(self.target)
        # added by Longyue
        assert num_sents == len(self.source_hist)

        if num_sents % self.batch_size == 0:
            self.blocks = num_sents / self.batch_size
        else:
            self.blocks = num_sents / self.batch_size + 1

    def test_print(self):
        print 'this is a test!'

    def get_iterator(self):
        for i in range(self.blocks):
            x = self.source[i*self.batch_size: (i+1)*self.batch_size]

            # added by Longyue
            x_hist = self.source_hist[i * self.batch_size: (i + 1) * self.batch_size]

            y = self.target[i*self.batch_size: (i+1)*self.batch_size]

            #modified by Longyue
            batch = self._create_padded_batch(x, x_hist, self.hist_len, y)
            yield batch


    def _create_padded_batch(self, x, x_hist, hist_len, y):

        # x_hist: (sent_num, hist_len, sent_len)

        mx = numpy.minimum(self.seq_len_src, max([len(xx) for xx in x])) + 1
        my = numpy.minimum(self.seq_len_trg, max([len(xx) for xx in y])) + 1

        batch_size = len(x)

        X = numpy.zeros((batch_size, mx), dtype='int64')
        Y = numpy.zeros((batch_size, my), dtype='int64')
        Xmask = numpy.zeros((batch_size, mx), dtype='float32')
        Ymask = numpy.zeros((batch_size, my), dtype='float32')

        for idx in range(len(x)):
            X[idx, :len(x[idx])] = x[idx]
            Xmask[idx, :len(x[idx])] = 1.
            if len(x[idx]) < mx:
                X[idx, len(x[idx]):] = self.eos_id
                Xmask[idx, len(x[idx])] = 1.

        for idx in range(len(y)):
            Y[idx,:len(y[idx])] = y[idx]
            Ymask[idx,:len(y[idx])] = 1.
            if len(y[idx]) < my:
                Y[idx, len(y[idx]):] = self.eos_id
                Ymask[idx, len(y[idx])] = 1.

        # added by Longyue
        lengths_hist = [[len(xx) for xx in s] for s in x_hist]
        mx_hist = numpy.minimum(self.seq_len_src, max([j for i in lengths_hist for j in i])) + 1
        X_hist = numpy.zeros((batch_size, hist_len, mx_hist), dtype='int64')
        Xmask_hist = numpy.zeros((batch_size, hist_len, mx_hist), dtype='float32')

        # print 'x_hist:',x_hist

        for idx, xx_hist in enumerate(x_hist): #idx=sent_id
            # print xx_hist
            for idx_hist in range(len(xx_hist)):
                X_hist[idx, idx_hist, :len(x_hist[idx][idx_hist])] = x_hist[idx][idx_hist]
                Xmask_hist[idx, idx_hist, :len(x_hist[idx][idx_hist])] = 1.
                if len(x_hist[idx][idx_hist]) < mx:
                    X_hist[idx, idx_hist, len(x_hist[idx][idx_hist]):] = self.eos_id
                    Xmask_hist[idx, idx_hist, len(x_hist[idx][idx_hist])] = 1.

        # print 'X_hist', X_hist
        # print 'Xmask_hist', Xmask_hist
        return X, Xmask, X_hist, Xmask_hist, Y, Ymask


    def _get_dict(self):

        if os.path.isfile(self.vocab_src):
            src_dict = pkl.load(open(self.vocab_src, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(self.vocab_src))

        if os.path.isfile(self.vocab_trg):
            trg_dict = pkl.load(open(self.vocab_trg, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(self.vocab_trg))

        return src_dict, trg_dict


    def _get_sentence_pairs(self):

        if os.path.isfile(self.train_src):
            f_src = open(self.train_src, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_src))

        if os.path.isfile(self.train_trg):
            f_trg = open(self.train_trg, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_trg))

        # added by Longyue
        if os.path.isfile(self.train_src_hist):
            f_src_hist = open(self.train_src_hist, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_src_hist))


        source = []
        target = []

        # added by Longyue
        source_hist = []

        for l_src, l_src_hist, l_trg in zip(f_src, f_src_hist, f_trg):
            src_words = l_src.strip().split()
            #src_words.append(self.eos_token)

            trg_words = l_trg.strip().split()
            #trg_words.append(self.eos_token)

            # added by Longyue
            src_words_hist_list = l_src_hist.strip().split('####')
            # print src_words_hist_list

            if len(src_words) == 0 or len(trg_words) == 0:
                continue

            if len(src_words) > self.seq_len_src or len(trg_words) > self.seq_len_trg:
                continue

            src_ids = [self.src_dict[w] if w in self.src_dict else self.unk_id for w in src_words]
            trg_ids = [self.trg_dict[w] if w in self.trg_dict else self.unk_id for w in trg_words]

            source.append(src_ids)
            target.append(trg_ids)

            # added by Longyue
            sent_list = []
            for sent in src_words_hist_list:
                src_hist_words = sent.strip().split()
                if src_hist_words == ['NULL']:
                    src_hist_ids = []
                else: src_hist_ids = [self.src_dict[w] if w in self.src_dict else self.unk_id for w in src_hist_words]
                sent_list.append(src_hist_ids)
            # print 'sent_list:', sent_list
            source_hist.append(sent_list)

        f_src.close()
        f_trg.close()

        #added by Longyue
        f_src_hist.close()

        #modified by Longyue
        return source, source_hist, target


    def _sort_by_k_batches(self, source, source_hist, target):

        bs = self.batch_size * self.sort_k_batches
        num_sents = len(source)
        assert num_sents == len(target)

        #added by Longyue
        assert num_sents == len(source_hist)

        if num_sents % bs == 0:
            blocks = num_sents / bs
        else:
            blocks = num_sents / bs + 1

        sort_source = []
        #added by Longyue
        sort_source_hist = []
        sort_target = []
        for i in range(blocks):
            tmp_src = numpy.asarray(source[i*bs:(i+1)*bs])
            # added by Longyue
            tmp_src_hist = numpy.asarray(source_hist[i * bs:(i + 1) * bs])
            tmp_trg = numpy.asarray(target[i*bs:(i+1)*bs])
            #modified by Longyue
            lens = numpy.asarray([map(len, tmp_src), map(len, tmp_src_hist), map(len, tmp_trg)])
            orders = numpy.argsort(lens[-1])
            for idx in orders:
                sort_source.append(tmp_src[idx])
                #added by Longyue
                sort_source_hist.append(tmp_src_hist[idx])
                sort_target.append(tmp_trg[idx])

        return sort_source, sort_source_hist, sort_target


def get_devtest_stream(data_type='valid', input_file=None, **kwards):

    if data_type == 'valid':
        data_file = kwards.pop('valid_src')
        data_file_hist = kwards.pop('valid_src_hist')
    elif data_type == 'test':
        if input_file is None:
            data_file = kwards.pop('test_src')
        else:
            data_file = input_file
        # added by Longyue
        data_file_hist = kwards.pop('test_src_hist')
    else:
        logger.error('wrong datatype, which must be one of valid or test')

    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')
    vocab_src = kwards.pop('vocab_src')

    dataset = TextFile(files=[data_file],
                       dictionary=pkl.load(open(vocab_src, 'rb')),
                       level='word',
                       unk_token=unk_token,
                       bos_token=None,
                       eos_token=eos_token)

    dev_stream = DataStream(dataset)

    # added by Longyue
    hist_len = 3
    dev_stream_hist=[]
    for idx in range(hist_len):
        dataset_hist = TextFile(files=[data_file_hist+str(idx)],
                           dictionary=pkl.load(open(vocab_src, 'rb')),
                           level='word',
                           unk_token=unk_token,
                           bos_token=None,
                           eos_token=eos_token)

        dev_stream_hist.append(DataStream(dataset_hist))

    dev_stream_hist_combine = []
    for d_s in dev_stream_hist:
        for item in d_s.get_epoch_iterator():
            dev_stream_hist_combine.append(item)

    item_len = len(dev_stream_hist_combine)
    dev_stream_hist_split = []
    for i in range(item_len / hist_len):
        tmp = []
        for j in range(hist_len):
            tmp.append(dev_stream_hist_combine[i + item_len / hist_len * j])
            dev_stream_hist_split.append(tmp)

    dev_stream_hist_split = tuple(dev_stream_hist_split)
    return dev_stream, dev_stream_hist_split


# added by Zhaopeng Tu
def get_stream(input_file, vocab_file, **kwards):
    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')

    dataset = TextFile(files=[input_file],
                       dictionary=pkl.load(open(vocab_file, 'rb')),
                       level='word',
                       unk_token=unk_token,
                       bos_token=None,
                       eos_token=eos_token)

    stream = DataStream(dataset)

    return stream

############## test read training data ##############
if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="get_config_search_coverage",
                        help="Prototype config to use for config")
    # added by Zhaopeng Tu, 2016-05-12
    parser.add_argument("--state", help="State to use")
    # added by Zhaopeng Tu, 2016-07-14
    parser.add_argument("--start", type=int, default=0, help="Iterations to start")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    configuration = getattr(configurations, args.proto)()
    ds = DStream(**configuration)

    for x, x_mask, x_hist, x_mask_hist, y, y_mask in ds.get_iterator():
        print 'x_hist', x_hist
        print 'x_hist', x_hist.shape

############## test read testing data ##############
# if __name__ == '__main__':
#     # Get the arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--proto", default="get_config_search_coverage",
#                         help="Prototype config to use for config")
#     # added by Zhaopeng Tu, 2016-05-12
#     parser.add_argument("--state", help="State to use")
#     # added by Zhaopeng Tu, 2016-07-14
#     parser.add_argument("--start", type=int, default=0, help="Iterations to start")
#     args = parser.parse_args()
#
#     logger = logging.getLogger(__name__)
#
#     configuration = getattr(configurations, args.proto)()
#
#     vs, vs_hist = get_devtest_stream(data_type='valid', input_file=None, **configuration)
#
#     print vs, vs_hist
#
#
#     for sent, sent_hist in zip(vs.get_epoch_iterator(), vs_hist):
#         print sent
#         print sent_hist
#         sent_hist_2 = tuple(sent_hist)
#         max_len = max([len(s[0]) for s in sent_hist_2])
#         print max_len
#         all = []
#         inp_mask_hist = numpy.zeros((3, max_len), dtype='float32')
#         for idx, s_hist in enumerate(sent_hist):
#             print 'print', s_hist
#             tmp = s_hist[0] + [0] * (max_len - len(s_hist[0]))
#             inp_mask_hist[idx, :len(s_hist[0])] = 1.
#             all.append(tmp)
#         print 'all', all
#         print 'mask', inp_mask_hist
#         print