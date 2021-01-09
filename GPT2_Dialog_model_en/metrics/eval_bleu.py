from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


smooth = SmoothingFunction()


def eval_bleu(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: corpus bleu score
    """
    bleu = corpus_bleu(ref, pred, smoothing_function=smooth.method1)
    if not 0 <= bleu <= 1:
        bleu = 0.0
    return bleu
