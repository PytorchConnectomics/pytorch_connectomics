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
from .cutblur import CutBlur
from .cutnoise import CutNoise
from .mixup import MixupAugmentor

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
           'CutBlur',
           'CutNoise',
           'MixupAugmentor',
           'TestAugmentor']


def build_train_augmentor(cfg, keep_uncropped=False, keep_non_smoothed=False):
    # The two arguments, keep_uncropped and keep_non_smoothed, are used only
    # for debugging, which are False by defaults and can not be adjusted
    # in the config files.
    aug_list = []
    #1. rotate
    if cfg.AUGMENTOR.ROTATE.ENABLED:
        aug_list.append(Rotate(p=cfg.AUGMENTOR.ROTATE.P))

    #2. rescale
    if cfg.AUGMENTOR.RESCALE.ENABLED:
        aug_list.append(Rescale(p=cfg.AUGMENTOR.RESCALE.P))

    #3. flip
    if cfg.AUGMENTOR.FLIP.ENABLED:
        aug_list.append(Flip(p=cfg.AUGMENTOR.FLIP.P, 
                             do_ztrans=cfg.AUGMENTOR.FLIP.DO_ZTRANS))

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
    if cfg.AUGMENTOR.MISSINGSECTION.ENABLED and not cfg.DATASET.DO_2D:
            aug_list.append(MissingSection(p=cfg.AUGMENTOR.MISSINGSECTION.P, 
                                           num_sections=cfg.AUGMENTOR.MISSINGSECTION.NUM_SECTION))

    #8. misalignment
    if cfg.AUGMENTOR.MISALIGNMENT.ENABLED and not cfg.DATASET.DO_2D:
            aug_list.append(MisAlignment(p=cfg.AUGMENTOR.MISALIGNMENT.P, 
                                         displacement=cfg.AUGMENTOR.MISALIGNMENT.DISPLACEMENT,
                                         rotate_ratio=cfg.AUGMENTOR.MISALIGNMENT.ROTATE_RATIO))
    #9. motion-blur
    if cfg.AUGMENTOR.MOTIONBLUR.ENABLED:
        aug_list.append(MotionBlur(p=cfg.AUGMENTOR.MOTIONBLUR.P, 
                                   sections=cfg.AUGMENTOR.MOTIONBLUR.SECTIONS, 
                                   kernel_size=cfg.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE))

    #10. cut-blur
    if cfg.AUGMENTOR.CUTBLUR.ENABLED:
        aug_list.append(CutBlur(p=cfg.AUGMENTOR.CUTBLUR.P, 
                                length_ratio=cfg.AUGMENTOR.CUTBLUR.LENGTH_RATIO, 
                                down_ratio_min=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN,
                                down_ratio_max=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX,
                                downsample_z=cfg.AUGMENTOR.CUTBLUR.DOWNSAMPLE_Z))

    #11. cut-noise
    if cfg.AUGMENTOR.CUTNOISE.ENABLED:
        aug_list.append(CutNoise(p=cfg.AUGMENTOR.CUTNOISE.P, 
                                length_ratio=cfg.AUGMENTOR.CUTNOISE.LENGTH_RATIO, 
                                scale=cfg.AUGMENTOR.CUTNOISE.SCALE))

    augmentor = Compose(aug_list, input_size=cfg.MODEL.INPUT_SIZE, smooth=cfg.AUGMENTOR.SMOOTH,
                        keep_uncropped=keep_uncropped, keep_non_smoothed=keep_non_smoothed)

    return augmentor
