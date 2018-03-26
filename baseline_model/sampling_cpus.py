# sampling: Sampler and BleuValidator
#from __future__ import print_function
import numpy
import argparse
import pprint
import os
import cPickle as pkl
import subprocess
import logging
import time
import re
import configurations_0
from search import BeamSearch
from nmt import EncoderDecoder
from stream import get_devtest_stream
from multiprocessing import Process, Queue

logger = logging.getLogger(__name__)


def get_dict(vocab_file, eos_token):

    if os.path.isfile(vocab_file):
        ddict = pkl.load(open(vocab_file, 'rb'))
    else:
        logger.error("file [{}] do not exist".format(vocab_file))

    iddict = dict()
    for kk, vv in ddict.iteritems():
        iddict[vv] = kk

    iddict[0] = eos_token

    return ddict, iddict


def idx_to_word(seq, ivocab, unk_token):

    return " ".join([ivocab.get(idx, unk_token) for idx in seq])


def translate_model(queue, rqueue, pid, configuration, normalize):

    rng = numpy.random.RandomState(1234)
    enc_dec = EncoderDecoder(rng, **configuration)
    enc_dec.build_sampler()
    enc_dec.load(path=configuration['saveto_best'])
    search_model = BeamSearch(enc_dec=enc_dec, \
                              beam_size=configuration['beam_size'], \
                              maxlen=3*configuration['seq_len_src'], \
                              stochastic=False,
                              configuration=configuration)

    def _translate(seq):
        outputs, scores = search_model.apply(numpy.array(seq).T)

        if normalize:
            lengths = numpy.array([len(s) for s in outputs])
            scores = scores / lengths
        sidx = numpy.argmin(scores)

        return outputs[sidx][:-1]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print pid, '-', idx

        seq = _translate(x)
        rqueue.put((idx, seq))

    return


# decoding with multi-cpus
def decoding_multi_cpus(src_file, trg_file, out_file, n_process, configuration):

    unk_token = configuration['unk_token']
    eos_token = configuration['eos_token']
    vocab_src = configuration['vocab_src']
    vocab_trg = configuration['vocab_trg']
    normalize = configuration['normalized_bleu']
    bleu_script = configuration['bleu_script']
    res_to_sgm = configuration['res_to_sgm']
    test_src = src_file
    test_ref = trg_file

    if src_file is None or trg_file is None:
        test_src = configuration['valid_src']
        test_ref = configuration['valid_trg']

    dict_src, idict_src = get_dict(vocab_src, eos_token)
    dict_trg, idict_trg = get_dict(vocab_trg, eos_token)

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(target=translate_model, args=(queue, rqueue, midx, configuration, normalize))
        processes[midx].start()


    def _send_jobs(src_file):
        data_stream = get_devtest_stream(data_type='test', input_file=src_file, **configuration)
        idx = 0
        for sent in data_stream.get_epoch_iterator():
            queue.put((idx, sent))
            idx += 1
        return idx


    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)


    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            if numpy.mod(idx+1, 100) == 0:
                logger.info("Translated {} lines of test set ...".format(idx+1))
        return trans


    logger.info("Begin decoding ...")
    val_start_time = time.time()
    n_samples = _send_jobs(test_src)
    trans = _retrieve_jobs(n_samples)
    _finish_processes()

    fout = open(out_file, 'w')
    for res in trans:
        res = idx_to_word(res, idict_trg, unk_token)
        if res.strip() == '':
            res = unk_token
        fout.write(res + '\n')
    fout.close()


    logger.info("Decoding took {} minutes".format(float(time.time() - val_start_time) / 60.))
    logger.info("Evaluate ...")

    cmd_res_to_sgm = [res_to_sgm, 'tst', out_file, '>', out_file+'.sgm']
    cmd_bleu_cmd = ['perl', bleu_script, \
                    '-r', test_ref+'.sgm', \
                    '-s', test_src+'.sgm', \
                    '-t', out_file+'.sgm', \
                    '>', test_src+'.eval']

    logger.info('covert result to sgm')
    subprocess.check_call(" ".join(cmd_res_to_sgm), shell=True)
    logger.info('compute bleu score')
    subprocess.check_call(" ".join(cmd_bleu_cmd), shell=True)

    fin = open(test_src+'.eval', 'rU')
    out = re.search('BLEU score = [-.0-9]+', fin.readlines()[7])
    fin.close()

    bleu_score = float(out.group()[13:])
    logger.info("Done")

    return bleu_score


if __name__=='__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="get_config_search_coverage",
                        help="Prototype config to use for config")
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('trans', type=str)
    parser.add_argument('-p', type=int, default=5)
    args = parser.parse_args()

    configuration = getattr(configurations_0, args.proto)()
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    test_bleu = decoding_multi_cpus(args.source, args.target, args.trans, args.p, configuration)
    logger.info('test bleu %.4f' %test_bleu)


