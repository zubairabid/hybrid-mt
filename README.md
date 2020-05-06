# Hybrid Machine Translation

- Souvik Banerjee (20171094)
- Zubair Abid (20171076)

## Instructions

### Data Setup

Downloads data and requisite preprocessing scripts. Preprocesses the data.

```bash
bash datasetup.sh
```

### Statistical MT

Clones the requisite repositories, trains a system, runs an evaluation and returns the score

```bash
bash baselinesmt.sh
```

### Baseline NMT

Clones the repo, does further preprocessing, trains a model, evaluates and returns a score

```bash
bash baselinenmt.sh
```

### UVR-NMT

Constructs lookup table, trains a model, evaluates, and returns a score.

TODO streamline
