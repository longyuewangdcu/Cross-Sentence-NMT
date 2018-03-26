# configurations for rnnsearch with coverage
def get_config_search_coverage():

    config = {}

    # added by Zhaopeng Tu, 2016-11-08
    # coverage penalty in GNMT
    # alpha in the paper
    config['length_penalty_factor'] = 0.0
    # beta in the paper
    config['coverage_penalty_factor'] = 0.0
    # pruning
    config['with_decoding_pruning'] = False
    # beamsize in the paper
    config['decoding_pruning_beam'] = 3

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = False
    # added by Zhaopeng Tu, 2016-06-09
    config['with_attention'] = True
    # added by Zhaopeng Tu, 2016-07-29
    config['output_kbest'] = False

    # added by Zhaopeng Tu, 2016-05-02
    # ACL 2016: Modeling Coverage for Neural Machine Translation
    # configurations for coverage model
    config['with_coverage'] = False
    # the coverage_dim for linguistic coverage is always 1
    config['coverage_dim'] = 100
    # coverage type: 'linguistic' or 'neural'
    config['coverage_type'] = 'neural' 
    # max value of fertility, the value of N in the paper
    config['max_fertility'] = 2

    # added by Zhaopeng Tu, 2016-05-30
    # TACL 2017: Context Gates for Neural Machine Translation
    # configurations for context gate
    config['with_context_gate'] = False

    # added by Zhaopeng Tu, 2016-07-11
    # AAAI 2017: Neural Machine Translation with Reconstruction
    # the reconstruction work
    config['with_reconstruction'] = False
    config['reconstruction_weight'] = 1.

    # Sequences longer than this will be deleted
    config['seq_len_src'] = 40
    config['seq_len_trg'] = 40

    # Number of hidden units in GRU/LSTM
    config['nhids_src'] = 50
    config['nhids_trg'] = 50

    # Dimension of the word embedding matrix
    config['nembed_src'] = 31
    config['nembed_trg'] = 31

    # Batch size of train data
    config['batch_size'] = 20

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 20

    # BeamSize
    config['beam_size'] = 10

    # Where to save model
    config['saveto'] = 'D:/MyWork/workspace/path/model.npz'
    config['saveto_best'] = 'D:/MyWork/workspace/path/model_best.npz'

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.5

    # Maxout, set maxout_part=1 to turn off
    config['maxout_part'] = 1

    # vocabulary size, include '</S>'
    config['src_vocab_size'] = 3001
    config['trg_vocab_size'] = 3001

    # Special tokens and indexes
    config['unk_id'] = 1
    config['unk_token'] = '<UNK>'
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'

    # Root directory for dataset
    datadir = 'D:/MyWork/workspace/NMT-ZPTU/data/toydata/'

    # added by Zhaopeng Tu, 2016-07-21
    config['replace_unk'] = False
    config['unk_dict'] = datadir + 'unk_dict'

    # Vocabularies
    config['vocab_src'] = datadir + 'vocab_src.pkl'
    config['vocab_trg'] = datadir + 'vocab_trg.pkl'

    # Datasets
    config['train_src'] = datadir + 'train_src.shuffle'
    config['train_trg'] = datadir + 'train_trg.shuffle'
    config['valid_src'] = datadir + 'valid_src'
    config['valid_trg'] = datadir + 'valid_trg'
    config['valid_out'] = datadir + 'valid_out'

    # Bleu script that will be used
    config['bleu_script'] = datadir + 'mteval-v11b.pl'
    config['res_to_sgm'] = datadir + 'plain2sgm.py'

    # Maxmum number of epoch
    config['finish_after'] = 20

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 50

    # Sample frequence
    config['sample_freq'] = 50
    # Hook samples
    config['hook_samples'] = 3

    # # Valid frequence
    # config['valid_freq'] = 100
    # config['valid_freq_fine'] = 50
    #
    # # Start bleu validation after this many updates
    # config['val_burn_in'] = 1000
    # config['val_burn_in_fine'] = 1500

    # Valid frequence
    config['valid_freq'] = 10000  # less than val_burn_in_fine
    config['valid_freq_fine'] = 5000  # more than val_burn_in_fine

    # Start bleu validation after this many updates
    config['val_burn_in'] = 10000  # less than it, no need validate
    config['val_burn_in_fine'] = 100000  # control which freq to be used

    # GRU, LSTM
    config['method'] = 'GRU'

    # Gradient clipping
    config['clip_c'] = 1.

    return config

