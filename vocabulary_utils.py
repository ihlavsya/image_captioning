"""utils for building vocabularies"""
import nltk
from pycocotools.coco import COCO
import gensim

import settings


def build_word2vec(filename):
    """Build a simple vocabulary wrapper."""
    coco = COCO(filename)
    annotation_ids = coco.anns.keys()
    corpus = []
    start_token = "<start>"
    end_token = "<end>"
    num_layers = 0
    for i, annotation_id in enumerate(annotation_ids):
        tokens = [start_token]
        caption = str(coco.anns[annotation_id]["caption"])
        tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokens.append(end_token)
        corpus.append(tokens)

        if num_layers < len(tokens):
            num_layers = len(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(annotation_ids)))

    tokens = ["<start>", "The", "<unk>", "of", "flowers", "<end>"]
    corpus.append(tokens)
    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        corpus,
        size=settings.EMBED_SIZE,  # 300
        window=5,
        min_count=1,
        workers=5)
    # you can try add again unk this way "dataset1/annotations/train.json"
    # it is in wv.vocab["char"].count
    return model, num_layers
