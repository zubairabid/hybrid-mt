#!/bin/bash
echo "Script to hopefully install moses, and then setup with data"
# Setting up main work directory
mkdir baselinesmt
cd baselinesmt
base_dir=`pwd`
corpus=$base_dir"/data/wmt-14"
moses=$base_dir"/mosesdecoder"
giza=$base_dir"/giza-pp"
lm=$base_dir"/lm"
working=$base_dir"/working"
binmod=$working"/binarised-model"

echo "############################"
echo "Downloading moses and giza++"
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/moses-smt/giza-pp.git

echo "Installing giza++"
cd giza-pp
make
cd $moses
mkdir tools
cp $giza/GIZA++-v2/GIZA++ $giza/GIZA++-v2/snt2cooc.out $giza/mkcls-v2/mkcls tools
cd $base_dir

echo "Installing Boost"
# TODO


echo "Installing moses itself"
cd $moses
./bjam -j4

echo "######################"
echo "Preprocessing the data"

mkdir $corpus/tokenized
mkdir $corpus/model
mkdir $corpus/clean

echo "Tokenising"
$moses/scripts/tokenizer/tokenizer.perl -l en < $corpus/europarl-v7.fr-en.en > $corpus/tokenized/europarl-v7.fr-en.tok.en
$moses/scripts/tokenizer/tokenizer.perl -l fr < $corpus/europarl-v7.fr-en.fr > $corpus/tokenized/europarl-v7.fr-en.tok.fr

echo "Truecasinf"
$moses/scripts/recaser/train-truecaser.perl --model $corpus/model/truecase-model.en --corpus $corpus/tokenized/europarl-v7.fr-en.tok.en
$moses/scripts/recaser/train-truecaser.perl --model $corpus/model/truecase-model.fr --corpus $corpus/tokenized/europarl-v7.fr-en.tok.fr

$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.en < $corpus/tokenized/europarl-v7.fr-en.tok.en > $corpus/tokenized/europarl-v7.fr-en.true.en
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.fr < $corpus/tokenized/europarl-v7.fr-en.tok.fr > $corpus/tokenized/europarl-v7.fr-en.true.fr

echo "Cleaning"
$moses/scripts/training/clean-corpus-n.perl $corpus/tokenized/europarl-v7.fr-en/true en fr $corpus/clean/europarl-v7.fr-en.clean 2 80

echo "#######################"
echo "Language model training"

mkdir $lm
$moses/bin/lmplz -o 3 < $corpus/tokenized/europarl-v7.fr-en.true.fr > $lm/europarl-v7.fr-en.arpa.fr

echo "binarising target language LM"
$moses/bin/build_binary $lm/europarl-v7.fr-en.arpa.fr $lm/europarl-v7.fr-en.blm.fr

echo "###################"
echo "Training and Tuning"

mkdir $working
cd $working
echo "Initial training"
nohup nice $moses/scripts/training/train-model.perl --root-dir train -corpus $corpus/clean/europarl-v7.fr-en.clean -f en -e fr -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:3:$lm/europarl-v7.fr-en.blm.fr:8 -external-bin-dir $moses/tools >& training.out &

echo "Downloading tuning corpus"
mkdir $corpus/tune
wget http://www.statmt.org/wmt12/dev.tgz
tar -zxvf dev.tgz

echo "Preprocessing tune data"
$moses/scripts/tokenizer/tokenizer.perl -l en < $corpus/tune/news-test2008.en > $corpus/tune/news-test2008.tok.en
$moses/scripts/tokenizer/tokenizer.perl -l fr < $corpus/tune/news-test2008.fr > $corpus/tune/news-test2008.tok.fr
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.en < $corpus/tune/news-test2008.tok.en > $corpus/tune/news-test2008.true.en
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.fr < $corpus/tunetokenized/news-test2008.tok.fr > $corpus/tune/news-test2008.true.fr

cd $working
echo "tuning being done"
nohup nice $moses/scripts/training/mert-moses.pl $corpus/tune/news-test2008.true.en $corpus/tune/news-test2008.true.fr $moses/bin/moses $workin/train/model/moses.ini --mertdir $moses/bin/ &> mert.out &

# Test
echo "#########################"
echo "Testing with newstest2011"
cd $corpus/tune
echo "Preprocessing test data"
$moses/scripts/tokenizer/tokenizer.perl -l en < $corpus/tune/newstest2011.en > $corpus/tune/newstest2011.tok.en
$moses/scripts/tokenizer/tokenizer.perl -l fr < $corpus/tune/newstest2011.fr > $corpus/tune/newstest2011.tok.fr
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.en < $corpus/tune/newstest2011.tok.en > $corpus/tune/newstest2011.true.en
$moses/scripts/recaser/truecase.perl --model $corpus/model/truecase-model.fr < $corpus/tunetokenized/newstest2011.tok.fr > $corpus/tune/newstest2011.true.fr

cd $working
echo "Filtering model for test set"
$moses/scripts/training/filter-model-given-input.pl filtered-newstest2011 mert-work/moses.ini $corpus/tune/newstest2011.true.en -Binarizer $moses/bin/processPhraseTableMin

nohup nice $moses/bin/moses -f $working/filtered-newstest2011/moses.ini < $corpus/tune/newstest2011.true.en > $working/newstest2011.translated.fr 2> $working/newstest2011.out $moses/scripts/generic/multi-bleu.perl -lc $corpus/tune/newstest2011.true.fr < $working/newstest2011.translated.fr



