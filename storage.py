"""storage for constant setting for initialization and models/training"""
from torchvision import transforms


class Storage():
    """storage for constant setting for initialization and models/training"""
    CAPTIONS_PER_IMAGE = 5
    EMBED_SIZE = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TEST_WV_WRAPPER_FILENAME = "test_models_weights/word_vectors_wrapper.pkl"
    DATA_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    BASE_HYPERPARAMS_DICT = {
        "embed_size": EMBED_SIZE,
        "hidden_size": EMBED_SIZE,
        "learning_rate": 0.001,
        "gamma": 0.5,
        "step_size": 1
    }
    BASE_JSON = {
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                "id": 3,
                "name": "Attribution-NonCommercial-NoDerivs License"
            },
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-sa/2.0/",
                "id": 5,
                "name": "Attribution-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nd/2.0/",
                "id": 6,
                "name": "Attribution-NoDerivs License"
            },
            {
                "url": "http://flickr.com/commons/usage/",
                "id": 7,
                "name": "No known copyright restrictions"
            },
            {
                "url": "http://www.usa.gov/copyright.shtml",
                "id": 8,
                "name": "United States Government Work"
            }
        ]
    }
