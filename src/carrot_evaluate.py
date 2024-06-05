import evaluate
import sacrebleu
import numpy as np

def reference_based_evaluate(preds, labels):
    Sacrebleu = sacrebleu.corpus_bleu(preds, labels).score
    CHRF = sacrebleu.corpus_chrf(preds, labels).score
    Rouge = evaluate.load("rouge").compute(predictions=preds, references=labels)['rougeL']
    BERT = evaluate.load("bertscore").compute(predictions=preds, references=labels,
                    model_type='bert-base-uncased', lang='en', verbose=True)['f1']
    print (f"Sacrebleu: {Sacrebleu}, CHRF: {CHRF}, Rouge: {np.mean(Rouge)*100}, BERT: {np.mean(BERT)*100}")
    return Sacrebleu, CHRF,  np.mean(Rouge)*100, np.mean(BERT)*100