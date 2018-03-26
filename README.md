# Cross-Sentence-NMT v1.0
Cross Sentence Neural Machine Translation

In translation, considering the document as a whole can help to resolve ambiguities and inconsistencies. We propose a cross-sentence context-aware approach to investigate the influence of historical contextual information on the performance of NMT. If you use the code, please cite <a href="http://www.aclweb.org/anthology/D17-1301">our paper</a>:

<pre><code>@InProceedings{Wang:2017:EMNLP,
  author    = {Wang, Longyue and Tu, Zhaopeng and Way, Andy and Liu Qun},
  title     = {Exploiting Cross-Sentence Context for Neural Machine Translation},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  year      = {2017},
}</code></pre>

This is an re-implemented version of [LC_NMT](http://www.aclweb.org/anthology/D17-1301). Currently, we only release the final best model ``+Initenc+dec+Gating Auxi'', which combines all the sub-models in the paper. This model is based on the a Theano-based RNNSearch (https://github.com/tuzhaopeng/NMT). 

For any comments or questions, please  email <a href="mailto:vincentwang0229@gmail.com">Longyue Wang</a> and <a href="mailto:tuzhaopeng@gmail.com">Zhaopeng Tu</a>.

Verification
--------------------------

The cross-sentence NMT models have been implemented in both Nematus (<a href="http://www.aclweb.org/anthology/D17-1301">LC-NMT</a>) and the new Theano-based RNNSearch (<a href="https://github.com/longyuewangdcu/Cross-Sentence-NMT">Cross-Sentence-NMT</a>>) architectures. For both versions, we obtained consistent performance.

Some following work has verified the improvement of our model. Please read [Learning to Remember Translation History with a Continuous Cache](https://arxiv.org/pdf/1711.09367.pdf) for more comparisons.

How to Run?
--------------------------

**1, Pre-processing**

We need to process train/dev/test data to create corresponding history data. Assume that the number of history sentence is 2 (hist_len = 2). The example is:

The orignal format of a document in the train file is:

<pre><code>And of course , we all share the same adaptive imperatives .
We 're all born . We all bring our children into the world .
We go through initiation rites .
We have to deal with the inexorable separation of death , so it shouldn 't surprise us that we all sing , we all dance , we all have art .</code></pre>

We need to create a new file which contains corresponding history sentences for each line in orignal train file. The format should be:

<pre><code>NULL####NULL
NULL####And of course , we all share the same adaptive imperatives .
And of course , we all share the same adaptive imperatives .####We 're all born . We all bring our children into the world .
We 're all born . We all bring our children into the world .####We go through initiation rites .</code></pre>

where the delimiter "####" is sentence boundary. Actually, you could DIY your input format by changing stram.py.

**2, Training and Testing**

2.0 Run shuffle.py and prepare.py to prepare the training data;

2.1 Set the NMT configurations in configurations.py;

2.2 Run train.sh to start training;

2.3 Run test.sh for decoding.

**3, To**

To-do List
--------------------------

1, Upload the experimental results on various domains;

2, Adding new models using target-side information.