import nltk
from nltk.translate.bleu_score import corpus_bleu
import sys

if len(sys.argv) != 3:
    print("Format: python {0} <prediction file> <test file>".format(sys.argv[0]))
else:
    with open(sys.argv[1]) as f:
        pred = f.read().split('\n')

    with open(sys.argv[2]) as f:
        og = f.read().split('\n')

    og_split = []
    pred_split = []
    for sentence in og:
        og_split.append(sentence.split(' '))

    for sentence in pred:
        pred_split.append(sentence.split(' '))

    bleu = corpus_bleu(og_split, pred_split, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7)
    print("Final BLEU Score: ",bleu)

