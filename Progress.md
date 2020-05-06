## Progress

### Data and cleaning

- Got WMT '14  EN-FR data
- General preprocessing: cleaning, tokenization
- Split into train-dev-test sets, for further tasks

### Baselines

#### Neural MT

- Implemented Seq2seq with attention. Code works for smaller subsets of data.

- Code is currently crashing due to insufficient compute

  ##### Test Runs

  *Configuration*: 1 Nvidia 1080Ti/2080Ti (it varied), 50 Epochs, batch size 64

  - `ENC_EMB_DIM, DEC_EMB_DIM = 256`, `ENC_HID_DIM, DEC_HID_DIM = 512` Program is terminated without any messages
  - `ENC_EMB_DIM, DEC_EMB_DIM = 128`, `ENC_HID_DIM, DEC_HID_DIM = 256` Program is terminated, CUDA explicit out of memory error
  - `ENC_EMB_DIM, DEC_EMB_DIM = 64`, `ENC_HID_DIM, DEC_HID_DIM = 128` Program runs, subject to further configuration on data size
  - `1606178 train.json lines` Program is terminated without any messages
  - `200000 train.json lines` Program is terminated, CUDA explicit out of memory error
  - `20000 train.json lines` Program runs. **Average BLEU Score: 3.1338220773474965e-162**
  - `100000 train.json lines` Program is terminated, CUDA explicit out of memory error
  - `50000 train.json lines`  Program is terminated, CUDA explicit out of memory error

  Now with batch size 16, because something went wrong?? Also changed the code for tokenisation and actual preprocessing, and smoothing on evaluation.

  - `ENC_EMB_DIM, DEC_EMB_DIM = 64`, `ENC_HID_DIM, DEC_HID_DIM = 128` Program runs, subject to further configuration on data size
  - `20000 train.json lines` Program runs. 
    - `smoothing method 1:` **Average BLEU Score: 1.7215e-4**
    - `smoothing method 2:` **Average BLEU Score: 7.2408e-4**
    - `smoothing method 3:` **Average BLEU Score: 3.4227e-4**
    - `smoothing method 4:` **Average BLEU Score: 5.42315e-4**
    - `smoothing method 5:` **Average BLEU Score: 2.5625e-4**
    - `smoothing method 7:` **Average BLEU Score: 7.6057e-4**

- 100,000 train.json, some configuration: 1.8481e-4

#### Statistical MT

- Setup Moses to work on a computer. About 5 hours.
- Got a working SMT system for FR-EN with KenLM language model (IRSTLM posing problems)

### Not-baselines

#### Neural MT

- Got data
- Started work on NMT with visual rep (WIP)

#### Statistical MT

- We were planning to use Joshua, might be infeasible. Looking at other options currently, like Stanford's Phrasal

## Results

# Baseline SMT

# Baseline NMT

# Baseline OpenNMT

- **0.282776** Kind of scam. Running on only 3391 sentences, due to errors in 
  the library.

# UVR-NMT




































