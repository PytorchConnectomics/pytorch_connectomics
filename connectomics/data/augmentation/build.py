from yacs.config import CfgNode

from .composition import Compose
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
from .copy_paste import CopyPasteAugmentor

def build_train_augmentor(cfg: CfgNode, keep_uncropped: bool = False, keep_non_smoothed: bool = False):
    r"""Build the training augmentor based on the options specified in the configuration
    file.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        keep_uncropped (bool): keep uncropped data in the output. Default: `False`
        keep_non_smoothed (bool): keep the masks before smoothing in the output. Default: `False`

    Note:
        The two arguments, keep_uncropped and keep_non_smoothed, are used only for debugging,
        which are `False` by defaults and can not be adjusted in the config file.
    """
    if not cfg.AUGMENTOR.ENABLED: # no data augmentation
        return None

    aug_list = []

    names = cfg.AUGMENTOR.ADDITIONAL_TARGETS_NAME
    types = cfg.AUGMENTOR.ADDITIONAL_TARGETS_TYPE
    if names is None:
        additional_targets = None
    else:
        assert len(names) == len(types)
        additional_targets = {}
        for i in range(len(names)):
            additional_targets[names[i]] = types[i]

    #1. rotate
    rotate_aug = cfg.AUGMENTOR.ROTATE
    if rotate_aug.ENABLED:
        aug_list.append(
            Rotate(rot90=rotate_aug.ROT90, p=rotate_aug.P,
                   additional_targets=additional_targets,
                   skip_targets=rotate_aug.SKIP))

    #2. rescale
    rescale_aug = cfg.AUGMENTOR.RESCALE
    if rescale_aug.ENABLED:
        aug_list.append(
            Rescale(p=rescale_aug.P, fix_aspect=rescale_aug.FIX_ASPECT,
                    additional_targets=additional_targets,
                    skip_targets=rescale_aug.SKIP))

    #3. flip
    flip_aug = cfg.AUGMENTOR.FLIP
    if flip_aug.ENABLED:
        aug_list.append(
            Flip(do_ztrans=flip_aug.DO_ZTRANS, p=flip_aug.P,
                 additional_targets=additional_targets,
                 skip_targets=flip_aug.SKIP))

    #4. elastic
    elastic_aug = cfg.AUGMENTOR.ELASTIC
    if elastic_aug.ENABLED:
        aug_list.append(
            Elastic(alpha=elastic_aug.ALPHA,
                    sigma=elastic_aug.SIGMA,
                    p=elastic_aug.P,
                    additional_targets=additional_targets,
                    skip_targets=elastic_aug.SKIP))

    #5. grayscale
    grayscale_aug = cfg.AUGMENTOR.GRAYSCALE
    if grayscale_aug.ENABLED:
        aug_list.append(
            Grayscale(p=grayscale_aug.P,
                      additional_targets=additional_targets,
                      skip_targets=grayscale_aug.SKIP))

    #6. missingparts
    missingparts_aug = cfg.AUGMENTOR.MISSINGPARTS
    if missingparts_aug.ENABLED:
        aug_list.append(
            MissingParts(iterations=missingparts_aug.ITER,
                         p=missingparts_aug.P,
                         additional_targets=additional_targets,
                         skip_targets=missingparts_aug.SKIP))

    #7. missingsection
    missingsection_aug = cfg.AUGMENTOR.MISSINGSECTION
    if missingsection_aug.ENABLED and not cfg.DATASET.DO_2D:
            aug_list.append(
                MissingSection(
                    num_sections=missingsection_aug.NUM_SECTION,
                    p=missingsection_aug.P,
                    additional_targets=additional_targets,
                    skip_targets=missingsection_aug.SKIP))

    #8. misalignment
    misalignment_aug = cfg.AUGMENTOR.MISALIGNMENT
    if misalignment_aug.ENABLED and not cfg.DATASET.DO_2D:
            aug_list.append(
                MisAlignment(
                    displacement=misalignment_aug.DISPLACEMENT,
                    rotate_ratio=misalignment_aug.ROTATE_RATIO,
                    p=misalignment_aug.P,
                    additional_targets=additional_targets,
                    skip_targets=misalignment_aug.SKIP))

    #9. motion-blur
    motionblur_aug = cfg.AUGMENTOR.MOTIONBLUR
    if motionblur_aug.ENABLED:
        aug_list.append(
            MotionBlur(
                sections=motionblur_aug.SECTIONS,
                kernel_size=motionblur_aug.KERNEL_SIZE,
                p=motionblur_aug.P,
                additional_targets=additional_targets,
                skip_targets=motionblur_aug.SKIP))

    #10. cut-blur
    cutblur_aug = cfg.AUGMENTOR.CUTBLUR
    if cutblur_aug.ENABLED:
        aug_list.append(
            CutBlur(length_ratio=cutblur_aug.LENGTH_RATIO,
                    down_ratio_min=cutblur_aug.DOWN_RATIO_MIN,
                    down_ratio_max=cutblur_aug.DOWN_RATIO_MAX,
                    downsample_z=cutblur_aug.DOWNSAMPLE_Z,
                    p=cutblur_aug.P,
                    additional_targets=additional_targets,
                    skip_targets=cutblur_aug.SKIP))

    #11. cut-noise
    cutnoise_aug = cfg.AUGMENTOR.CUTNOISE
    if cutnoise_aug.ENABLED:
        aug_list.append(
            CutNoise(length_ratio=cutnoise_aug.LENGTH_RATIO,
                     scale=cutnoise_aug.SCALE,
                     p=cutnoise_aug.P,
                     additional_targets=additional_targets,
                     skip_targets=cutnoise_aug.SKIP))

    #12. copy-paste
    copypaste_aug = cfg.AUGMENTOR.COPYPASTE
    if copypaste_aug.ENABLED:
        aug_list.append(
            CopyPasteAugmentor(aug_thres=copypaste_aug.AUG_THRES,
                     p=copypaste_aug.P,
                     additional_targets=additional_targets,
                     skip_targets=copypaste_aug.SKIP))

    # compose the list of transforms
    augmentor = Compose(transforms=aug_list,
                        input_size=cfg.MODEL.INPUT_SIZE,
                        smooth=cfg.AUGMENTOR.SMOOTH,
                        keep_uncropped=keep_uncropped,
                        keep_non_smoothed=keep_non_smoothed,
                        additional_targets=additional_targets)

    return augmentor


def build_ssl_augmentor(cfg):
    r"""Build the data augmentor for semi-supervised learning.
    """
    aug_list = []

    #1. rotate
    if cfg.AUGMENTOR.ROTATE.ENABLED:
        aug_list.append(
            Rotate(rot90=True,
                   p=cfg.AUGMENTOR.ROTATE.P))

    #2. flip
    if cfg.AUGMENTOR.FLIP.ENABLED:
        aug_list.append(
            Flip(do_ztrans=cfg.AUGMENTOR.FLIP.DO_ZTRANS,
                 p=cfg.AUGMENTOR.FLIP.P))

    #3. grayscale
    if cfg.AUGMENTOR.GRAYSCALE.ENABLED:
        aug_list.append(
            Grayscale(p=cfg.AUGMENTOR.GRAYSCALE.P))

    #4. missingparts
    if cfg.AUGMENTOR.MISSINGPARTS.ENABLED:
        aug_list.append(
            MissingParts(p=cfg.AUGMENTOR.MISSINGPARTS.P))

    #5. motion-blur
    if cfg.AUGMENTOR.MOTIONBLUR.ENABLED:
        aug_list.append(
            MotionBlur(
                sections=cfg.AUGMENTOR.MOTIONBLUR.SECTIONS,
                kernel_size=cfg.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE,
                p=cfg.AUGMENTOR.MOTIONBLUR.P))

    #6. cut-blur
    if cfg.AUGMENTOR.CUTBLUR.ENABLED:
        aug_list.append(
            CutBlur(length_ratio=cfg.AUGMENTOR.CUTBLUR.LENGTH_RATIO,
                    down_ratio_min=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN,
                    down_ratio_max=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX,
                    downsample_z=cfg.AUGMENTOR.CUTBLUR.DOWNSAMPLE_Z,
                    p=cfg.AUGMENTOR.CUTBLUR.P))

    #7. cut-noise
    if cfg.AUGMENTOR.CUTNOISE.ENABLED:
        aug_list.append(
            CutNoise(length_ratio=cfg.AUGMENTOR.CUTNOISE.LENGTH_RATIO,
                     scale=cfg.AUGMENTOR.CUTNOISE.SCALE,
                     p=cfg.AUGMENTOR.CUTNOISE.P))

    return Compose(transforms=aug_list,
                   input_size=cfg.MODEL.INPUT_SIZE,
                   smooth=cfg.AUGMENTOR.SMOOTH,
                   additional_targets=None)
