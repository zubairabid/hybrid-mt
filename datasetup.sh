#!/bin/bash
# Adapted from https://github.com/cooelf/UVR-NMT/blob/master/prepare-wmt-en2de.sh

echo "Welcome to the setup tool!"
project_root=`pwd`

echo "----------------------------------------------------------"
echo "Downloading and setting up preprocessing scripts"
echo "----------------------------------------------------------"
script_dir=$project_root/scripts
moses=$script_dir/mosesdecoder
moses_scripts=$moses/scripts

mkdir -p $script_dir
cd $script_dir

echo "Cloning the Moses repository for preprocessing scripts"
git clone https://github.com/moses-smt/mosesdecoder.git
echo "Cloning the Subword NMT repository for BPE preprocessing"
git clone https://github.com/rsennrich/subword-nmt.git

echo "----------------------------------------------------------"
echo "Downloading all the data fles needed"
echo "----------------------------------------------------------"

data_dir=$project_root/data
multi30k=$data_dir/multi30k
tmp_multi30k=$multi30k/data/task1/tok
wmt_dir=$data_dir/downloads
wmt_url="https://www.statmt.org/europarl/v7/fr-en.tgz"
wmt_file="fr-en.tgz"

mkdir -p wmt_dir

echo "Downloading and extracting the WMT '14 EN-FR data"
cd $wmt_dir
if [ -f $wmt_file ]; then
    echo "$wmt_file already exists, skipping download"
else
    wget $wmt_url
    if [ -f $wmt_file ]; then
        echo "$wmt_url successfully downloaded."
    else
        echo "$wmt_url not successfully downloaded."
        exit -1
    fi
    tar -zxvf $wmt_file
fi

cd $data_dir
echo "Cloning the Multi30k dataset"
git clone --recursive https://github.com/multi30k/dataset.git multi30k

cd $project_root    

echo "----------------------------------------------------------"
echo "Pre-preprocessing the data"
echo "----------------------------------------------------------"

TOKENIZER=$moses_scripts/tokenizer/tokenizer.perl
CLEAN=$moses_scripts/training/clean-corpus-n.perl
NORM_PUNC=$moses_scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$moses_scripts/tokenizer/remove-non-printing-char.perl

outdir=$data_dir/wmt14_en_fr
tmp=$outdir/tmp

src=en
tgt=fr

mkdir -p tmp

echo "Normalising, removing non-printing chars, and tokenising"
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    f="$wmt_dir/europarl-v7.fr-en"
    cat $f.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 2 -a -l $l >> $tmp/train.tags.$lang.tok.$l
done

echo "Splitting data into train, test, validation sets"
for l in $src $tgt; do
    awk '{if (NR%667 == 0 && NR%2 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%667 == 0 && NR%2 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/test.$l
    awk '{if (NR%667 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

bpe_root=$script_dir/subword-nmt/subword-nmt
bpe_code=$outdir/code
bpe_tokens=40000

train=$tmp/train.fr-en

echo "Learning BPE Values from ${train}"
# Concatenating train+multi30k into one train file
rm -f $train
for l in $src $tgt; do
    cat $tmp/train.$l >> $train
    cat $tmp_multi30k/train.lc.norm.tok.$l >> $train
done

python $bpe_root/learn_bpe.py -s $bpe_tokens < $train > $bpe_code

echo "Using the BPE"

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $bpe_root/apply_bpe.py -c $bpe_code < $tmp/$f > $tmp/bpe.$f
    done
    python $bpe_root/apply_bpe.py -c $bpe_code < $tmp_multi30k/"train.lc.norm.tok.en" > $outdir/"bpe.multi30k.en"
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $outdir/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $outdir/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $outdir/test.$L
done
