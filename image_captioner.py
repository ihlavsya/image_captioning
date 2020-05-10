"""ImageCaptioner is model which encapsulates
two other models: encoder and decoder in chain manner"""
import torch.nn as nn


class ImageCaptioner(nn.Module):
    """ImageCaptioner is model which encapsulates
    two other models: encoder and decoder in chain manner"""

    def __init__(self, encoder, decoder):
        """init model"""
        super(ImageCaptioner, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, images, captions, lengths):
        """perform forwardpass"""
        features = self.__encoder(images)
        outputs = self.__decoder(features, captions, lengths)
        return outputs

    def get_encoder_state_dict(self):
        """property for getting encoder state_dict"""
        encoder_state_dict = self.__encoder.state_dict()
        return encoder_state_dict

    def get_decoder_state_dict(self):
        """property for getting decoder state_dict"""
        decoder_state_dict = self.__decoder.state_dict()
        return decoder_state_dict
