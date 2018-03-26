# rnn encoder-decoder for machine translation
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import logging
import os
from models import LookupTable, LogisticRegression, BidirectionalEncoder, Encoder, Decoder, InverseDecoder
from utils import Dropout, concatenate
from algorithm import adadelta, grad_clip
# added by Zhaopeng Tu, 2016-07-30
# for debugging NaN
# from theano.compile.nanguardmode import NanGuardMode

logger = logging.getLogger(__name__)


class EncoderDecoder(object):

    def __init__(self, rng, **kwargs):
        self.n_in_src = kwargs.pop('nembed_src')
        self.n_in_trg = kwargs.pop('nembed_trg')
        self.n_hids_src = kwargs.pop('nhids_src')
        self.n_hids_trg = kwargs.pop('nhids_trg')
        self.src_vocab_size = kwargs.pop('src_vocab_size')
        self.trg_vocab_size = kwargs.pop('trg_vocab_size')
        self.method = kwargs.pop('method')
        self.dropout = kwargs.pop('dropout')
        self.maxout_part = kwargs.pop('maxout_part')
        self.path = kwargs.pop('saveto')
        self.clip_c = kwargs.pop('clip_c')
        self.rng = rng
        self.trng = RandomStreams(rng.randint(1e5))

        # added by  Zhaopeng  Tu, 2016-06-09
        self.with_attention = kwargs.pop('with_attention')

        # added by Zhaopeng Tu, 2016-04-29
        self.with_coverage = kwargs.pop('with_coverage')
        self.coverage_dim = kwargs.pop('coverage_dim')
        self.coverage_type = kwargs.pop('coverage_type')
        self.max_fertility = kwargs.pop('max_fertility')
        if self.coverage_type is 'linguistic':
            # make sure the dimension of linguistic coverage is always 1
            self.coverage_dim = 1
        
        # added by Zhaopeng Tu, 2016-05-30
        self.with_context_gate = kwargs.pop('with_context_gate')

        self.params = []
        self.layers = []

        self.table_src = LookupTable(self.rng, self.src_vocab_size, self.n_in_src, name='table_src')
        self.layers.append(self.table_src)

        self.encoder = BidirectionalEncoder(self.rng, self.n_in_src, self.n_hids_src, self.table_src, name='birnn_encoder')
        self.layers.append(self.encoder)

        # added by Longyue
        self.encoder_hist_1 = Encoder(self.rng, self.n_in_src, self.n_hids_src, self.table_src, name='rnn_encoder_hist_1')
        self.layers.append(self.encoder_hist_1)
        self.encoder_hist_2 = Encoder(self.rng, self.n_hids_src, self.n_hids_src, self.table_src, name='rnn_encoder_hist_2')
        self.layers.append(self.encoder_hist_2)

        self.table_trg = LookupTable(self.rng, self.trg_vocab_size, self.n_in_trg, name='table_trg')
        self.layers.append(self.table_trg)

        self.decoder = Decoder(self.rng, self.n_in_trg, self.n_hids_trg, 2*self.n_hids_src, self.n_hids_src, \
                               # added by Zhaopeng Tu, 2016-06-09
                               with_attention=self.with_attention, \
                               # added by Zhaopeng Tu, 2016-04-29
                               with_coverage=self.with_coverage, coverage_dim=self.coverage_dim, coverage_type=self.coverage_type, max_fertility=self.max_fertility, \
                               # added by Zhaopeng Tu, 2016-05-30
                               with_context_gate=self.with_context_gate, \
                               maxout_part=self.maxout_part, name='rnn_decoder')
        self.layers.append(self.decoder)
        self.logistic_layer = LogisticRegression(self.rng, self.n_in_trg, self.trg_vocab_size)
        self.layers.append(self.logistic_layer)

        # added by Zhaopeng Tu, 2016-07-12
        # for reconstruction
        self.with_reconstruction = kwargs.pop('with_reconstruction')
        if self.with_reconstruction:
            # added by Zhaopeng Tu, 2016-07-27
            self.reconstruction_weight = kwargs.pop('reconstruction_weight')
            # note the source and target sides are reversed
            self.inverse_decoder = InverseDecoder(self.rng, self.n_in_src, 2*self.n_hids_src, self.n_hids_trg, \
                                   # added by Zhaopeng Tu, 2016-06-09
                                   with_attention=self.with_attention, \
                                   maxout_part=self.maxout_part, name='rnn_inverse_decoder')
            self.layers.append(self.inverse_decoder)
            
            self.srng = RandomStreams(rng.randint(1e5))
            self.inverse_logistic_layer = LogisticRegression(self.rng, self.n_in_src, self.src_vocab_size, name='inverse_LR')
            self.layers.append(self.inverse_logistic_layer)

        for layer in self.layers:
            self.params.extend(layer.params)


    def build_trainer(self, src, src_mask, src_hist, src_hist_mask, trg, trg_mask, ite):

        # added by Longyue
        # checked by Zhaopeng: sentence dim = n_steps, hist_len, batch_size (4, 3, 25)
        # hist = (bath_size, sent_num, sent_len) --.T-->
        # hist = (sent_len, sent_num, bath_size) --lookup table-->
        # (sent_len, sent_num, bath_size, word_emb) --reshape-->
        # (sent_len, sent_num*bath_size, word_emb) --word-level rnn-->
        # (sent_len, sent_num*bath_size, hidden_size) --reshape-->
        # (sent_len, sent_num, bath_size, hidden_size) --[-1]-->
        # (sent_num, bath_size, hidden_size) --sent-level rnn-->
        # (sent_num, bath_size, hidden_size) --[-1]-->
        # (bath_size, hidden_size) = cross-sent context vector

        annotations_1 = self.encoder_hist_1.apply_1(src_hist, src_hist_mask)
        annotations_1 = annotations_1[-1] # get last hidden states
        annotations_2 = self.encoder_hist_2.apply_2(annotations_1)
        annotations_3 = annotations_2[-1] # get last hidden states

        #modified by Longyue
        annotations = self.encoder.apply(src, src_mask, annotations_3)
        # init_context = annotations[0, :, -self.n_hids_src:]
        # modification #1
        # mean pooling
        init_context = (annotations * src_mask[:, :, None]).sum(0) / src_mask.sum(0)[:, None]

        #added by Longyue
        init_context = concatenate([init_context, annotations_3], axis=annotations_3.ndim - 1)

        trg_emb = self.table_trg.apply(trg)
        trg_emb_shifted = T.zeros_like(trg_emb)
        trg_emb_shifted = T.set_subtensor(trg_emb_shifted[1:], trg_emb[:-1])
        # modified by Longyue
        hiddens, readout, alignment = self.decoder.run_pipeline(state_below=trg_emb_shifted,
                                            mask_below=trg_mask,
                                            init_context=init_context,
                                            c=annotations,
                                            c_mask=src_mask,
                                            hist=annotations_3)


        # apply dropout
        if self.dropout < 1.0:
            logger.info('Apply dropout with p = {}'.format(self.dropout))
            readout = Dropout(self.trng, readout, 1, self.dropout)

        p_y_given_x = self.logistic_layer.get_probs(readout)

        self.cost = self.logistic_layer.cost(p_y_given_x, trg, trg_mask) / trg.shape[1]

        # self.cost = theano.printing.Print('likilihood cost:')(self.cost)

        # added by Zhaopeng Tu, 2016-07-12
        # for reconstruction
        if self.with_reconstruction:
            # now hiddens is the annotations
            inverse_init_context = (hiddens * trg_mask[:, :, None]).sum(0) / trg_mask.sum(0)[:, None]

            src_emb = self.table_src.apply(src)
            src_emb_shifted = T.zeros_like(src_emb)
            src_emb_shifted = T.set_subtensor(src_emb_shifted[1:], src_emb[:-1])
            inverse_hiddens, inverse_readout, inverse_alignment = self.inverse_decoder.run_pipeline(state_below=src_emb_shifted,
                                                mask_below=src_mask,
                                                init_context=inverse_init_context,
                                                c=hiddens,
                                                c_mask=trg_mask)

            # apply dropout
            if self.dropout < 1.0:
                # logger.info('Apply dropout with p = {}'.format(self.dropout))
                inverse_readout = Dropout(self.srng, inverse_readout, 1, self.dropout)

            p_x_given_y = self.inverse_logistic_layer.get_probs(inverse_readout)

            self.reconstruction_cost = self.inverse_logistic_layer.cost(p_x_given_y, src, src_mask) / src.shape[1]

            # self.reconstruction_cost = theano.printing.Print('reconstructed cost:')(self.reconstruction_cost)
            self.cost += self.reconstruction_cost * self.reconstruction_weight
            

        self.L1 = sum(T.sum(abs(param)) for param in self.params)
        self.L2 = sum(T.sum(param ** 2) for param in self.params)

        params_regular = self.L1 * 1e-6 + self.L2 * 1e-6
        # params_regular = theano.printing.Print('params_regular:')(params_regular)

        # train cost
        train_cost = self.cost + params_regular

        # gradients
        grads = T.grad(train_cost, self.params)

        # apply gradient clipping here
        grads = grad_clip(grads, self.clip_c)

        # updates
        updates = adadelta(self.params, grads)

        # train function
        # modified by Longyue
        inps = [src, src_mask, src_hist, src_hist_mask, trg, trg_mask]

        self.train_fn = theano.function(inps, [train_cost], updates=updates, name='train_function')
        # self.train_fn = theano.function(inps, [train_cost], updates=updates, name='train_function', mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))



    def build_sampler(self):

        # added by Longyue
        x_hist = T.ltensor3()
        x_hist_mask = T.tensor3()
        annotations_1 = self.encoder_hist_1.apply_1(x_hist, x_hist_mask)
        annotations_1 = annotations_1[-1]
        annotations_2 = self.encoder_hist_2.apply_2(annotations_1)
        annotations_3 = annotations_2[-1]


        x = T.lmatrix()


        # Build Networks
        # src_mask is None
        c = self.encoder.apply(x, None, annotations_3)
        #init_context = ctx[0, :, -self.n_hids_src:]
        # mean pooling
        init_context = c.mean(0)

        # added by Longyue
        init_context = concatenate([init_context, annotations_3], axis=annotations_3.ndim - 1)

        init_state = self.decoder.create_init_state(init_context)

        outs = [init_state, c, annotations_3]
        if not self.with_attention:
            outs.append(init_context)

        # compile function
        print 'Building compile_init_state_and_context function ...'
        self.compile_init_and_context = theano.function([x, x_hist, x_hist_mask], outs,
                                                        name='compile_init_and_context')
        print 'Done'


        y = T.lvector()
        cur_state = T.matrix()
        # if it is the first word, emb should be all zero, and it is indicated by -1
        trg_emb = T.switch(y[:, None] < 0,
                           T.alloc(0., 1, self.n_in_trg),
                           self.table_trg.apply(y))

        # added by Zhaopeng Tu, 2016-06-09
        # for with_attention=False
        if self.with_attention and self.with_coverage:
            cov_before = T.tensor3()
            if self.coverage_type is 'linguistic':
                print 'Building compile_fertility ...'
                fertility = self.decoder._get_fertility(c)
                fertility = T.addbroadcast(fertility,1)
                self.compile_fertility = theano.function([c], [fertility], name='compile_fertility')
                print 'Done'
            else:
                fertility = None
        else:
            cov_before = None
            fertility = None

        # apply one step
        # modified by Zhaopeng Tu, 2016-04-29
        # [next_state, ctxs] = self.decoder.apply(state_below=trg_emb,
        results = self.decoder.apply(state_below=trg_emb,
                                     init_state=cur_state,
                                     # added by Zhaopeng Tu, 2016-06-09
                                     init_context=None if self.with_attention else init_context,
                                     c=c if self.with_attention else None,
                                     hist=annotations_3, # added by Longyue
                                     one_step=True,
                                     # added by Zhaopeng Tu, 2016-04-27
                                     cov_before=cov_before,
                                     fertility=fertility)
        next_state = results[0]
        if self.with_attention:
            ctxs, alignment = results[1], results[2]
            if self.with_coverage:
                cov = results[3]
        else:
            # if with_attention=False, we always use init_context as the source representation
            ctxs = init_context

        readout = self.decoder.readout(next_state, ctxs, trg_emb)

        # maxout
        if self.maxout_part > 1:
            readout = self.decoder.one_step_maxout(readout)

        # apply dropout
        if self.dropout < 1.0:
            readout = Dropout(self.trng, readout, 0, self.dropout)

        # compute the softmax probability
        next_probs = self.logistic_layer.get_probs(readout)

        # sample from softmax distribution to get the sample
        next_sample = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile function
        print 'Building compile_next_state_and_probs function ...'
        inps = [y, cur_state]
        if self.with_attention:
            inps.append(c)
        else:
            inps.append(init_context)

        # added by Longyue
        inps.append(annotations_3)

        outs = [next_probs, next_state, next_sample]
        # added by Zhaopeng Tu, 2016-06-09
        if self.with_attention:
            outs.append(alignment)
            # added by Zhaopeng Tu, 2016-04-29
            if self.with_coverage:
                inps.append(cov_before)
                if self.coverage_type is 'linguistic':
                    inps.append(fertility)
                outs.append(cov)

        self.compile_next_state_and_probs = theano.function(inps, outs, name='compile_next_state_and_probs')
        print 'Done'

        # added by Zhaopeng Tu, 2016-07-18
        # for reconstruction
        if self.with_reconstruction:
            # Build Networks
            # trg_mask is None
            inverse_c = T.tensor3()
            # mean pooling
            inverse_init_context = inverse_c.mean(0)

            inverse_init_state = self.inverse_decoder.create_init_state(inverse_init_context)

            outs = [inverse_init_state]
            if not self.with_attention:
                outs.append(inverse_init_context)

            # compile function
            print 'Building compile_inverse_init_state_and_context function ...'
            self.compile_inverse_init_and_context = theano.function([inverse_c], outs, name='compile_inverse_init_and_context')
            print 'Done'


            src = T.lvector()
            inverse_cur_state = T.matrix()
            trg_mask = T.matrix()
            # if it is the first word, emb should be all zero, and it is indicated by -1
            src_emb = T.switch(src[:, None] < 0,
                               T.alloc(0., 1, self.n_in_src),
                               self.table_src.apply(src))

            # apply one step
            # modified by Zhaopeng Tu, 2016-04-29
            inverse_results = self.inverse_decoder.apply(state_below=src_emb,
                                         init_state=inverse_cur_state,
                                         # added by Zhaopeng Tu, 2016-06-09
                                         init_context=None if self.with_attention else inverse_init_context,
                                         c=inverse_c if self.with_attention else None,
                                         c_mask=trg_mask,
                                         one_step=True)
            inverse_next_state = inverse_results[0]
            if self.with_attention:
                inverse_ctxs, inverse_alignment = inverse_results[1], inverse_results[2]
            else:
                # if with_attention=False, we always use init_context as the source representation
                inverse_ctxs = init_context

            inverse_readout = self.inverse_decoder.readout(inverse_next_state, inverse_ctxs, src_emb)

            # maxout
            if self.maxout_part > 1:
                inverse_readout = self.inverse_decoder.one_step_maxout(inverse_readout)

            # apply dropout
            if self.dropout < 1.0:
                inverse_readout = Dropout(self.srng, inverse_readout, 0, self.dropout)

            # compute the softmax probability
            inverse_next_probs = self.inverse_logistic_layer.get_probs(inverse_readout)

            # sample from softmax distribution to get the sample
            inverse_next_sample = self.srng.multinomial(pvals=inverse_next_probs).argmax(1)

            # compile function
            print 'Building compile_inverse_next_state_and_probs function ...'
            inps = [src, trg_mask, inverse_cur_state]
            if self.with_attention:
                inps.append(inverse_c)
            else:
                inps.append(inverse_init_context)
            outs = [inverse_next_probs, inverse_next_state, inverse_next_sample]
            # added by Zhaopeng Tu, 2016-06-09
            if self.with_attention:
                outs.append(inverse_alignment)

            self.compile_inverse_next_state_and_probs = theano.function(inps, outs, name='compile_inverse_next_state_and_probs')
            print 'Done'





    def save(self, path=None):
        if path is None:
            path = self.path
        filenpz = open(path, "w")
        val = dict([(value.name, value.get_value()) for index, value in enumerate(self.params)])
        logger.info("save the model {}".format(path))
        numpy.savez(path, **val)
        filenpz.close()


    def load(self, path=None):
        if path is None:
            path = self.path
        if os.path.isfile(path):
            logger.info("load params {}".format(path))
            val = numpy.load(path)
            for index, param in enumerate(self.params):
                logger.info('Loading {} with shape {}'.format(param.name, param.get_value(borrow=True).shape))
                if param.name not in val.keys():
                    logger.info('Adding new param {} with shape {}'.format(param.name, param.get_value(borrow=True).shape))
                    continue
                if param.get_value().shape != val[param.name].shape:
                    logger.info("Error: model param != load param shape {} != {}".format(\
                                        param.get_value().shape, val[param.name].shape))
                    raise Exception("loading params shape mismatch")
                else:
                    param.set_value(val[param.name], borrow=True)
        else:
            logger.error("file {} does not exist".format(path))
            self.save()
