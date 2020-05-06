#!/bin/bash

echo "Script to train and run moses."
echo "Requirements: Boost installed, Moses downloaded via ./datasetup.sh. Moses and GIZA++ will be installed (GIZA++ will also be downloaded)"

# Setting up main work directory
project_root=`pwd`
data_dir=$project_root/data
base_dir=$project_root/baselinesmt
moses=$project_root/scripts/mosesdecoder

giza=$base_dir/giza-pp
lm=$base_dir/lm
working=$base_dir/working
binmod=$working/binarised-model

echo "----------------------------------------------------------"
echo "Installing GIZA++ and Moses"
echo "----------------------------------------------------------"

mkdir $base_dir
cd $base_dir

echo "Downloading and installing GIZA++"
git clone https://github.com/moses-smt/giza-pp.git

cd $giza
make
mkdir -p $moses/tools
cp $giza/GIZA++-v2/GIZA++ $giza/GIZA++-v2/snt2cooc.out $giza/mkcls-v2/mkcls $moses/tools
cd $base_dir

echo "Installing moses itself"
cd $moses
./bjam -j4

echo "----------------------------------------------------------"
echo "Language model training"
echo "----------------------------------------------------------"

corpus=$data_dir/wmt14_en_fr

lm=$base_dir/lm

mkdir $lm
$moses/bin/lmplz -o 3 < $corpus/tmp/train.tags.en-fr.tok.fr > $lm/europarl-v7.fr-en.arpa.fr

echo "binarising target language LM"
$moses/bin/build_binary $lm/europarl-v7.fr-en.arpa.fr $lm/europarl-v7.fr-en.blm.fr

echo "----------------------------------------------------------"
echo "Training and Tuning"
echo "----------------------------------------------------------"

working=$base_dir/working

cd $working
echo "Initial training"
nohup nice $moses/scripts/training/train-model.perl --root-dir train -corpus $corpus/train -f en -e fr -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:3:$lm/europarl-v7.fr-en.blm.fr:8 -external-bin-dir $moses/tools >& training.out &

echo "Downloading tuning corpus"
mkdir $corpus/tune
cd $corpus/tune
wget http://www.statmt.org/wmt12/dev.tgz
tar -zxvf dev.tgz

echo "Preprocessing tune data"
$moses/scripts/tokenizer/tokenizer.perl -l en < $corpus/tune/dev/news-test2008.en > $corpus/tune/news-test2008.tok.en
$moses/scripts/tokenizer/tokenizer.perl -l fr < $corpus/tune/dev/news-test2008.fr > $corpus/tune/news-test2008.tok.fr
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.en < $corpus/tune/news-test2008.tok.en > $corpus/tune/news-test2008.true.en
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.fr < $corpus/tunetokenized/news-test2008.tok.fr > $corpus/tune/news-test2008.true.fr

cd $working
echo "tuning being done"
nohup nice $moses/scripts/training/mert-moses.pl $corpus/tune/news-test2008.true.en $corpus/tune/news-test2008.true.fr $moses/bin/moses $working/train/model/moses.ini --mertdir $moses/bin/ &> mert.out &

# Test
echo "----------------------------------------------------------"
echo "Testing with newstest2011"
echo "----------------------------------------------------------"
cd $corpus/tune
echo "Preprocessing test data"
$moses/scripts/tokenizer/tokenizer.perl -l en < $corpus/tune/dev/newstest2011.en > $corpus/tune/newstest2011.tok.en
$moses/scripts/tokenizer/tokenizer.perl -l fr < $corpus/tune/dev/newstest2011.fr > $corpus/tune/newstest2011.tok.fr
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.en < $corpus/tune/newstest2011.tok.en > $corpus/tune/newstest2011.true.en
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.fr < $corpus/tune/newstest2011.tok.fr > $corpus/tune/newstest2011.true.fr

cd $working
echo "Filtering model for test set"
$moses/scripts/training/filter-model-given-input.pl filtered-newstest2011 mert-work/moses.ini $corpus/tune/newstest2011.true.en -Binarizer $moses/bin/processPhraseTableMin

nohup nice $moses/bin/moses -f $working/filtered-newstest2011/moses.ini < $corpus/tune/newstest2011.true.en > $working/newstest2011.translated.fr 2> $working/newstest2011.out $moses/scripts/generic/multi-bleu.perl -lc $corpus/tune/newstest2011.true.fr < $working/newstest2011.translated.fr



