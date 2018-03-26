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
import copy
import configurations_0
from search import Reconstruct
from nmt import EncoderDecoder
from stream import get_stream

logger = logging.getLogger(__name__)


class Reconstructor(object):

    def __init__(self, reconstruction_model, test_src, test_trg, **kwards):
        self.reconstruction_model = reconstruction_model
        self.test_src = test_src
        self.test_trg = test_trg

        self.unk_token = kwards.pop('unk_token')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
       
        # added by Zhaopeng Tu, 2016-06-09
        self.with_attention = kwards.pop('with_attention')

        # added by Zhaopeng Tu, 2016-05-04
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')
        
        # added by Zhaopeng Tu, 2016-07-19
        self.with_reconstruction = kwards.pop('with_reconstruction')

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)


    def apply(self, output, verbose=False):
        fout = open(output, 'w')
            
        while 1:
            try:
                source = self.test_src.get_data()
                target = self.test_trg.get_data()
            except:
                break

            results = self.reconstruction_model.apply(numpy.array(source).T, numpy.array(target).T)
            reconstructed_source, reconstructed_score, alignment, reconstructed_alignment = results
            alignment = numpy.array(alignment).transpose()

            res = self._idx_to_word(reconstructed_source[:-1], self.idict_src)
            if res.strip() == '':
                res = self.unk_token

            print >> fout, res

            if verbose:
                source = source[0]
                target = target[0]
                logger.info("Input: {}".format(self._idx_to_word(source, self.idict_src)))
                logger.info("Target: {}".format(self._idx_to_word(target, self.idict_trg)))
                logger.info("Alignment: {}".format(alignment.tolist()))

                # added by Zhaopeng Tu, 2016-07-19
                if self.with_reconstruction:
                    logger.info("Reconstructed Input: {}".format(res))
                    logger.info("Inverse Alignment: {}".format(alignment.tolist()))



    def _get_dict(self, vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk, vv in ddict.iteritems():
            iddict[vv] = kk

        iddict[0] = self.eos_token

        return ddict, iddict

    # modified by Zhaopeng Tu, 2016-05-04
    # def _idx_to_word(self, seq, ivocab):
    def _idx_to_word(self, seq, ivocab, coverage=None):
        if coverage is None:
            return " ".join([ivocab.get(idx, self.unk_token) for idx in seq])
        else:
            output = []
            for i, [idx, ratio] in enumerate(zip(seq, coverage)):
                output.append('%s/%.2f' % (ivocab.get(idx, self.unk_token), ratio))
            return " ".join(output)


if __name__=='__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="get_config_search_coverage",
                        help="Prototype config to use for config")
    # added by Zhaopeng Tu, 2016-05-12
    parser.add_argument("--state", help="State to use")
    # added by Zhaopeng Tu, 2016-05-27
    parser.add_argument("--model", help="Model to use")
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('reconstruction', type=str)
    args = parser.parse_args()

    configuration = getattr(configurations_0, args.proto)()
    # added by Zhaopeng Tu, 2016-05-12
    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    rng = numpy.random.RandomState(1234)

    enc_dec = EncoderDecoder(rng, **configuration)
    enc_dec.build_sampler()

    # added by Zhaopeng Tu, 2016-05-27
    # options to use other trained models
    if args.model:
        enc_dec.load(path=args.model)
    else:
        enc_dec.load(path=configuration['saveto_best'])

    test_reconstruction = Reconstruct(enc_dec=enc_dec, 
                        configuration=configuration,
                        maxlen=3*configuration['seq_len_trg'], stochastic=True)

    test_src = get_stream(args.source, configuration['vocab_src'], **configuration)
    test_trg = get_stream(args.target, configuration['vocab_trg'], **configuration)
    reconstructor = Reconstructor(reconstruction_model=test_reconstruction, test_src=test_src, test_trg=test_trg, **configuration)

    reconstructor.apply(args.reconstruction, True)
