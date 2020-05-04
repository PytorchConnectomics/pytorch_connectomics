from .composition import Compose
from .augmentor import DataAugment
from .test_augmentor import TestAugmentor

# augmentation methods
from .warp import Elastic
from .grayscale import Grayscale
from .flip import Flip
from .rotation import Rotate
from .rescale import Rescale
from .misalign import MisAlignment
from .missing_section import MissingSection
from .missing_parts import MissingParts

__all__ = ['Compose',
           'DataAugment', 
           'Elastic',
           'Grayscale',
           'Rotate',
           'Rescale',
           'MisAlignment',
           'MissingSection',
           'MissingParts',
           'Flip',
           'TestAugmentor']


def build_train_augmentor(cfg):
    aug_list = []
    #1. rotate
    if cfg.AUGMENTOR.ROTATE:
        aug_list.append(Rotate(p=cfg.AUGMENTOR.ROTATE_P))
    #2. rescale
    if cfg.AUGMENTOR.RESCALE:
        aug_list.append(Rescale(p=cfg.AUGMENTOR.RESCALE_P))
    #3. flip
    if cfg.AUGMENTOR.FLIP:
        aug_list.append(Flip(p=cfg.AUGMENTOR.FLIP_P, do_ztrans=cfg.AUGMENTOR.FLIP_DO_ZTRANS))
    #4. elastic
    if cfg.AUGMENTOR.ELASTIC:
        aug_list.append(Elastic(alpha=cfg.AUGMENTOR.ELASTIC_ALPHA, sigma = cfg.AUGMENTOR.ELASTIC_SIGMA, p=cfg.AUGMENTOR.ELASTIC_P))
    #5. grayscale
    if cfg.AUGMENTOR.GRAYSCALE:
        aug_list.append(Grayscale(p=cfg.AUGMENTOR.GRAYSCALE_P))
    #6. missingparts
    if cfg.AUGMENTOR.MISSINGPARTS:
        aug_list.append(MissingParts(p=cfg.AUGMENTOR.MISSINGPARTS_P))
    #7. missingsection
    if cfg.AUGMENTOR.MISSINGSECTION:
        aug_list.append(MissingSection(p=cfg.AUGMENTOR.MISSINGSECTION_P))
    #8. misalignment
    if cfg.AUGMENTOR.MISALIGNMENT:
        aug_list.append(MisAlignment(p=cfg.AUGMENTOR.MISALIGNMENT_P, displacement=cfg.AUGMENTOR.MISALIGNMENT_DISPLACEMENT))

    augmentor = Compose(aug_list, input_size = cfg.MODEL.INPUT_SIZE)

    return augmentor
