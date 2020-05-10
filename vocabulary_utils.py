"""utils for building vocabularies"""
import pickle

import gensim
from pycocotools.coco import COCO
from nltk.tokenize import RegexpTokenizer

import contractions

from storage import Storage


def build_word2vec(filename):
    """Build a simple vocabulary wrapper."""
    coco = COCO(filename)
    annotation_ids = coco.anns.keys()
    corpus = []
    start_token = "<start>"
    end_token = "<end>"
    num_layers = 0
    tokenizer = RegexpTokenizer(r'\w+')
    for i, annotation_id in enumerate(annotation_ids):
        tokens = [start_token]
        caption = str(coco.anns[annotation_id]["caption"])
        caption = contractions.fix(caption)
        tokens.extend(tokenizer.tokenize(caption.lower()))
        tokens.append(end_token)
        corpus.append(tokens)

        if num_layers < len(tokens):
            num_layers = len(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions."
                  .format(i+1, len(annotation_ids)))

    fake_tokens = ["<start>", "The", "<unk>", "of", "flowers", "<end>"]
    corpus.append(fake_tokens)
    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        corpus,
        size=Storage.EMBED_SIZE,  # 300
        window=5,
        min_count=1,
        workers=5)
    # you can try add again unk this way "dataset1/annotations/train.json"
    # it is in wv.vocab["char"].count
    # yes, rewrite this please
    # try both methods
    return model, num_layers


def main():
    """run build_word2vec and save result"""
    captions_train_filename = "dataset1/annotations/train.json"
    model, num_layers = build_word2vec(captions_train_filename)
    word_vectors_wrapper = {"wv": model.wv, "num_layers": num_layers}

    filename = Storage.TEST_WV_WRAPPER_FILENAME
    with open(filename, "wb") as file:
        pickle.dump(word_vectors_wrapper, file,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
