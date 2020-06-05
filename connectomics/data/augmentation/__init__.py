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
from .motion_blur import MotionBlur

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
           'MotionBlur',
           'TestAugmentor']


def build_train_augmentor(cfg):
    aug_list = []
    #1. rotate
    if cfg.AUGMENTOR.ROTATE.ENABLED:
        aug_list.append(Rotate(p=cfg.AUGMENTOR.ROTATE.P))
    #2. rescale
    if cfg.AUGMENTOR.RESCALE.ENABLED:
        aug_list.append(Rescale(p=cfg.AUGMENTOR.RESCALE.P))
    #3. flip
    if cfg.AUGMENTOR.FLIP.ENABLED:
        aug_list.append(Flip(p=cfg.AUGMENTOR.FLIP.P, do_ztrans=cfg.AUGMENTOR.FLIP.DO_ZTRANS))
    #4. elastic
    if cfg.AUGMENTOR.ELASTIC.ENABLED:
        aug_list.append(Elastic(alpha=cfg.AUGMENTOR.ELASTIC.ALPHA, 
                                sigma = cfg.AUGMENTOR.ELASTIC.SIGMA, 
                                p=cfg.AUGMENTOR.ELASTIC.P))
    #5. grayscale
    if cfg.AUGMENTOR.GRAYSCALE.ENABLED:
        aug_list.append(Grayscale(p=cfg.AUGMENTOR.GRAYSCALE.P))
    #6. missingparts
    if cfg.AUGMENTOR.MISSINGPARTS.ENABLED:
        aug_list.append(MissingParts(p=cfg.AUGMENTOR.MISSINGPARTS.P))
    #7. missingsection
    if cfg.AUGMENTOR.MISSINGSECTION.ENABLED:
        aug_list.append(MissingSection(p=cfg.AUGMENTOR.MISSINGSECTION.P))
    #8. misalignment
    if cfg.AUGMENTOR.MISALIGNMENT.ENABLED:
        aug_list.append(MisAlignment(p=cfg.AUGMENTOR.MISALIGNMENT.P, 
                                     displacement=cfg.AUGMENTOR.MISALIGNMENT.DISPLACEMENT))
    #8. motion-blur
    if cfg.AUGMENTOR.MOTIONBLUR.ENABLED:
        aug_list.append(MotionBlur(p=cfg.AUGMENTOR.MOTIONBLUR.P, 
                                   sections=cfg.AUGMENTOR.MOTIONBLUR.SECTIONS, 
                                   kernel_size=cfg.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE))

    augmentor = Compose(aug_list, input_size = cfg.MODEL.INPUT_SIZE)

    return augmentor
