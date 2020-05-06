echo "Using OpenNMT"

project_root=`pwd`
opennmt=$project_root/opennmt

cd $opennmt
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd opennmt
pip install -r requirements.txt
cd $opennmt

data_dir=$project_root/data/wmt14_en_fr
open_prep=$opennmt/preprocessed
open_model=$opennmt/model

mkdir -p $open_prep

python preprocess.py -train_src $data_dir/train.en -train_tgt $data_dir/train.fr -valid_src $data_dir/valid.en -valid_tgt $data_dir/valid.fr -save_data $open_prep/prep 

python train.py -data $open_prep/prep -save_model $open_model -model -world_size 4 -gpu_ranks 0 1 2 3

python translate.py -model "${open_model}_step_100000.pt" -src $data_dir/test.en -output $data_dir/pred.fr -gpu 1 -replace_unk -verbose

python $project_root/scripts/bleu.py $data_dir/pred.fr $data_dir/test.fr
