TOOL_ROOT=/home/vwang/path

THEANO_FLAGS='device=gpu0,floatX=float32' python $TOOL_ROOT/sampling_2.py --beam $2 --model $1/model.npz /home/vwang/path/test_src /home/vwang/path/test_trg $1/test_baseline_out.$2 1>$1/test_baseline.log.$2 2>$1/test_baseline.err.$2

#head -1082 $1/test_out > $1/nist05_tran.txt
#tail -3021 $1/test_out | head -1664 > $1/nist06_tran.txt
#tail -1357 $1/test_out > $1/nist08_tran.txt

TOOL_ROOT=/home/vwang/workspaces/sample/data

$TOOL_ROOT/plain2sgm.py $1/test_baseline_out.$2 /home/vwang/path/test_src.sgm $1/test_baseline_out.sgm.$2
$TOOL_ROOT/mteval-v11b.pl -r /home/vwang/path/test_trg.sgm -s /home/vwang/data/path/test_src.sgm -t $1/test_baseline_out.sgm.$2 > $1/test_baseline_out.bleu.$2
 
#./data/plain2sgm.py $1/nist06_tran.txt data/nist06_src.sgm $1/nist06_tran.sgm
#./data/mteval-v11b.pl -r data/nist06_ref.sgm -s data/nist06_src.sgm -t $1/nist06_tran.sgm > $1/nist06_tran.bleu
 
#./data/plain2sgm.py $1/nist08_tran.txt data/nist08_src.sgm $1/nist08_tran.sgm
#./data/mteval-v11b.pl -r data/nist08_ref.sgm -s data/nist08_src.sgm -t $1/nist08_tran.sgm > $1/nist08_tran.bleu
 
