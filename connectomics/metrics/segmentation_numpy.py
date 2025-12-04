import numpy as np
import scipy.sparse as sparse
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from collections import namedtuple
from skimage.segmentation import relabel_sequential

matching_criteria = dict()

__all__ = ["jaccard", "adapted_rand", "instance_matching", "cremi_distance"]


def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]

    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.

    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error

    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)

    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # Validate shapes match
    if seg.shape != gt.shape:
        raise ValueError(
            f"seg and gt must have the same shape. "
            f"Got seg.shape={seg.shape}, gt.shape={gt.shape}"
        )

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = int(np.amax(segA)) + 1
    n_labels_B = int(np.amax(segB)) + 1

    ones_data = np.ones(n, int)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are


# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py


def voi(reconstruction, groundtruth, ignore_reconstruction=[], ignore_groundtruth=[0]):
    """Return the conditional entropies of the variation of information metric. [1]

    Let X be a reconstruction, and Y a ground truth labelling. The variation of
    information between the two is the sum of two conditional entropies:

        VI(X, Y) = H(X|Y) + H(Y|X).

    The first one, H(X|Y), is a measure of oversegmentation, the second one,
    H(Y|X), a measure of undersegmentation. These measures are referred to as
    the variation of information split or merge error, respectively.

    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg, ignore_gt : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        By default, only the label 0 in the ground truth will be ignored.

    Returns
    -------
    (split, merge) : float
        The variation of information split and merge error, i.e., H(X|Y) and H(Y|X)

    References
    ----------
    [1] Meila, M. (2007). Comparing clusterings - an information based
    distance. Journal of Multivariate Analysis 98, 873-895.
    """
    (hyxg, hxgy) = split_vi(reconstruction, groundtruth, ignore_reconstruction, ignore_groundtruth)
    return (hxgy, hyxg)


def split_vi(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return the symmetric conditional entropies associated with the VI.

    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.

    If y is None, x is assumed to be a contingency table.

    Parameters
    ----------
    x : np.ndarray
        Label field (int type) or contingency table (float). `x` is
        interpreted as a contingency table (summing to 1.0) if and only if `y`
        is not provided.
    y : np.ndarray of int, same shape as x, optional
        A label field to compare to `x`.
    ignore_x, ignore_y : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        Ignore 0-labeled points by default.

    Returns
    -------
    sv : np.ndarray of float, shape (2,)
        The conditional entropies of Y|X and X|Y.

    See Also
    --------
    vi
    """
    _, _, _, hxgy, hygx, _, _ = vi_tables(x, y, ignore_x, ignore_y)
    # false merges, false splits
    return np.array([hygx.sum(), hxgy.sum()])


def vi_tables(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return probability tables used for calculating VI.

    If y is None, x is assumed to be a contingency table.

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that may or may not sum to 1.
    ignore_x, ignore_y : list of int, optional
        Rows and columns (respectively) to ignore in the contingency table.
        These are labels that are not counted when evaluating VI.

    Returns
    -------
    pxy : sparse.csc_matrix of float
        The normalized contingency table.
    px, py, hxgy, hygx, lpygx, lpxgy : np.ndarray of float
        The proportions of each label in `x` and `y` (`px`, `py`), the
        per-segment conditional entropies of `x` given `y` and vice-versa, the
        per-segment conditional probability p log p.
    """
    if y is not None:
        pxy = contingency_table(x, y, ignore_x, ignore_y)
    else:
        cont = x
        total = float(cont.sum())
        # normalize, since it is an identity op if already done
        pxy = cont / total

    # Calculate probabilities
    px = np.array(pxy.sum(axis=1)).ravel()
    py = np.array(pxy.sum(axis=0)).ravel()
    # Remove zero rows/cols
    nzx = px.nonzero()[0]
    nzy = py.nonzero()[0]
    nzpx = px[nzx]
    nzpy = py[nzy]
    nzpxy = pxy[nzx, :][:, nzy]

    # Calculate log conditional probabilities and entropies
    lpygx = np.zeros(np.shape(px))
    lpygx[nzx] = xlogx(divide_rows(nzpxy, nzpx)).sum(axis=1).ravel()
    # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px * lpygx)  # \sum_x{p_x H(Y|X=x)} = H(Y|X)

    lpxgy = np.zeros(np.shape(py))
    lpxgy[nzy] = xlogx(divide_columns(nzpxy, nzpy)).sum(axis=0).ravel()
    hxgy = -(py * lpxgy)

    return [pxy] + list(map(np.asarray, [px, py, hxgy, hygx, lpygx, lpxgy]))


def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg : list of int, optional
        Values to ignore in `seg`. Voxels in `seg` having a value in this list
        will not contribute to the contingency table. (default: [0])
    ignore_gt : list of int, optional
        Values to ignore in `gt`. Voxels in `gt` having a value in this list
        will not contribute to the contingency table. (default: [0])
    norm : bool, optional
        Whether to normalize the table so that it sums to 1.

    Returns
    -------
    cont : scipy.sparse.csc_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `seg` and `j` in `gt`. (Or the proportion of such voxels
        if `norm=True`.)
    """
    segr = seg.ravel()
    gtr = gt.ravel()
    ignored = np.zeros(segr.shape, bool)
    data = np.ones(len(gtr))
    for i in ignore_seg:
        ignored[segr == i] = True
    for j in ignore_gt:
        ignored[gtr == j] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (segr, gtr))).tocsc()
    if norm:
        cont /= float(cont.sum())
    return cont


def divide_columns(matrix, row, in_place=False):
    """Divide each column of `matrix` by the corresponding element in `row`.

    The result is as follows: out[i, j] = matrix[i, j] / row[j]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (N,)
        The row dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if isinstance(out, (sparse.csc_matrix, sparse.csr_matrix)):
        if isinstance(out, sparse.csc_matrix):
            convert_to_csc = True
            out = out.tocsr()
        else:
            convert_to_csc = False
        row_repeated = np.take(row, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= row_repeated[nz]
        if convert_to_csc:
            out = out.tocsc()
    else:
        out /= row[np.newaxis, :]
    return out


def divide_rows(matrix, column, in_place=False):
    """Divide each row of `matrix` by the corresponding element in `column`.

    The result is as follows: out[i, j] = matrix[i, j] / column[i]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (M,)
        The column dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if isinstance(out, (sparse.csc_matrix, sparse.csr_matrix)):
        if type(out) == sparse.csr_matrix:  # noqa: E721
            convert_to_csr = True
            out = out.tocsc()
        else:
            convert_to_csr = False
        column_repeated = np.take(column, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= column_repeated[nz]
        if convert_to_csr:
            out = out.tocsr()
    else:
        out /= column[:, np.newaxis]
    return out


def xlogx(x, out=None, in_place=False):
    """Compute x * log_2(x).

    We define 0 * log_2(0) = 0

    Parameters
    ----------
    x : np.ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    out : same type as x (optional)
        If provided, use this array/matrix for the result.
    in_place : bool (optional, default False)
        Operate directly on x.

    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    if in_place:
        y = x
    elif out is None:
        y = x.copy()
    else:
        y = out
    if type(y) in [sparse.csc_matrix, sparse.csr_matrix]:
        z = y.data
    else:
        z = y
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y


# Functions for evaluating binary segmentation


def confusion_matrix(pred, gt, thres=0.5):
    """Calculate the confusion matrix given a probablility threshold in (0,1)."""
    TP = np.sum((gt == 1) & (pred > thres))
    FP = np.sum((gt == 0) & (pred > thres))
    TN = np.sum((gt == 0) & (pred <= thres))
    FN = np.sum((gt == 1) & (pred <= thres))
    return (TP, FP, TN, FN)


def jaccard(pred, gt, thres=[0.5]):
    """Evaluate the binary prediction at multiple thresholds using the Jaccard
    Index, which is also known as Intersection over Union (IoU). If the prediction
    is already binarized, different thresholds will result the same result.

    Args:
        pred (numpy.ndarray): foreground probability of shape :math:`(Z, Y, X)`.
        gt (numpy.ndarray): binary foreground label of shape identical to the prediction.
        thres (list): a list of probablility threshold in (0,1). Default: [0.5]

    Return:
        score (numpy.ndarray): a numpy array of shape :math:`(N, 4)`, where :math:`N` is the
        number of element(s) in the probability threshold list. The four scores in each line
        are foreground IoU, IoU, precision and recall, respectively.
    """
    score = np.zeros((len(thres), 4))
    for tid, t in enumerate(thres):
        assert 0.0 < t < 1.0, "The range of the threshold should be (0,1)."
        TP, FP, TN, FN = confusion_matrix(pred, gt, t)

        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        iou_fg = float(TP) / (TP + FP + FN)
        iou_bg = float(TN) / (TN + FP + FN)
        iou = (iou_fg + iou_bg) / 2.0
        score[tid] = np.array([iou_fg, iou, precision, recall])
    return score


def cremi_distance(pred, gt, resolution=(40.0, 4.0, 4.0)):
    """Compute the FP/FN statistics between predictions and ground truth as
    in the CREMI challenge (https://cremi.org/). Both inputs (pred, gt) need
    to be of the same size.
    """

    def count_false_positives(test_clefts_mask, truth_clefts_edt, threshold=200):
        mask1 = np.invert(test_clefts_mask)
        mask2 = truth_clefts_edt > threshold
        false_positives = truth_clefts_edt[np.logical_and(mask1, mask2)]
        return false_positives.size

    def count_false_negatives(truth_clefts_mask, test_clefts_edt, threshold=200):
        mask1 = np.invert(truth_clefts_mask)
        mask2 = test_clefts_edt > threshold
        false_negatives = test_clefts_edt[np.logical_and(mask1, mask2)]
        return false_negatives.size

    def acc_false_positives(test_clefts_mask, truth_clefts_edt):
        mask = np.invert(test_clefts_mask)
        false_positives = truth_clefts_edt[mask]
        stats = {
            "mean": np.mean(false_positives),
            "std": np.std(false_positives),
            "max": np.amax(false_positives),
            "count": false_positives.size,
            "median": np.median(false_positives),
        }
        return stats

    def acc_false_negatives(truth_clefts_mask, test_clefts_edt):
        mask = np.invert(truth_clefts_mask)
        false_negatives = test_clefts_edt[mask]
        stats = {
            "mean": np.mean(false_negatives),
            "std": np.std(false_negatives),
            "max": np.amax(false_negatives),
            "count": false_negatives.size,
            "median": np.median(false_negatives),
        }
        return stats

    def convert_dtype(data):
        data = data.astype(np.uint64)
        data[data == 0] = 0xFFFFFFFFFFFFFFFF
        return data

    test_clefts = convert_dtype(pred)
    truth_clefts = convert_dtype(gt)

    truth_clefts_invalid = truth_clefts == 0xFFFFFFFFFFFFFFFE

    test_clefts_mask = np.logical_or(test_clefts == 0xFFFFFFFFFFFFFFFF, truth_clefts_invalid)
    truth_clefts_mask = np.logical_or(truth_clefts == 0xFFFFFFFFFFFFFFFF, truth_clefts_invalid)

    print("EDT calculation in progress")
    test_clefts_edt = ndimage.distance_transform_edt(test_clefts_mask, sampling=resolution)
    truth_clefts_edt = ndimage.distance_transform_edt(truth_clefts_mask, sampling=resolution)

    false_positive_count = count_false_positives(test_clefts_mask, truth_clefts_edt)
    false_negative_count = count_false_negatives(truth_clefts_mask, test_clefts_edt)

    false_positive_stats = acc_false_positives(test_clefts_mask, truth_clefts_edt)
    false_negative_stats = acc_false_negatives(truth_clefts_mask, test_clefts_edt)

    print("Clefts Statistics")
    print("======")

    print("\tfalse positives: " + str(false_positive_count))
    print("\tfalse negatives: " + str(false_negative_count))

    print("\tdistance to ground truth: " + str(false_positive_stats))
    print("\tdistance to proposal    : " + str(false_negative_stats))

    return false_positive_stats["mean"], false_negative_stats["mean"]


# Code modified from https://github.com/stardist/stardist


# Copied from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def label_are_sequential(y):
    """returns true if y has only sequential labels from 1..."""
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))


def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)


def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError(
        "{label} must be an array of {integers}.".format(
            label="labels" if name is None else name,
            integers=("sequential " if check_sequential else "") + "non-negative integers",
        )
    )
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x, "x", True)
        _check_label_array(y, "y", True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)


def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _safe_divide(x, y, eps=1e-10):
    """computes a safe divide which returns 0 if y is zero"""
    if np.isscalar(x) and np.isscalar(y):
        return x / y if np.abs(y) > eps else 0.0
    else:
        out = np.zeros(np.broadcast(x, y).shape, np.float32)
        np.divide(x, y, out=out, where=np.abs(y) > eps)
        return out


def intersection_over_union(overlap):
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))


matching_criteria["iou"] = intersection_over_union


def intersection_over_true(overlap):
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pixels_true)


matching_criteria["iot"] = intersection_over_true


def intersection_over_pred(overlap):
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return _safe_divide(overlap, n_pixels_pred)


matching_criteria["iop"] = intersection_over_pred


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    # also known as "average precision" (?)
    # -> https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    # also known as "dice coefficient"
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def instance_matching(y_true, y_pred, thresh=0.5, criterion="iou", report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false negatives (fn)
    whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default IoU)
    report_matches: bool
        if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')

    Returns
    -------
    Matching object with different metrics as attributes

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true,5,axis = 0)

    >>> stats = instance_matching(y_true, y_pred)
    >>> print(stats)
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0, recall=0, accuracy=0, f1=0, n_true=1, n_pred=1, mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

    """
    _check_label_array(y_true, "y_true")
    _check_label_array(y_pred, "y_pred")
    y_true.shape == y_pred.shape or _raise(
        ValueError(
            "y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(
                y_true=y_true, y_pred=y_pred
            )
        )
    )
    criterion in matching_criteria or _raise(
        ValueError("Matching criterion '%s' not supported." % criterion)
    )
    if thresh is None:
        thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float, thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    map_rev_true = np.array(map_rev_true)
    map_rev_pred = np.array(map_rev_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignoring background
    scores = scores[1:, 1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2 * n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind, pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        # assert tp+fp == n_pred
        # assert tp+fn == n_true

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind, pred_ind][match_ok]) if not_trivial else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score = _safe_divide(sum_matched_score, n_true)
        panoptic_quality = _safe_divide(sum_matched_score, tp + fp / 2 + fn / 2)

        stats_dict = dict(
            criterion=criterion,
            thresh=thr,
            fp=fp,
            tp=tp,
            fn=fn,
            precision=precision(tp, fp, fn),
            recall=recall(tp, fp, fn),
            accuracy=accuracy(tp, fp, fn),
            f1=f1(tp, fp, fn),
            n_true=n_true,
            n_pred=n_pred,
            mean_true_score=mean_true_score,
            mean_matched_score=mean_matched_score,
            panoptic_quality=panoptic_quality,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update(
                    # int() to be json serializable
                    matched_pairs=tuple(
                        (int(map_rev_true[i]), int(map_rev_pred[j]))
                        for i, j in zip(1 + true_ind, 1 + pred_ind)
                    ),
                    matched_scores=tuple(scores[true_ind, pred_ind]),
                    matched_tps=tuple(map(int, np.flatnonzero(match_ok))),
                    pred_ids=tuple(map_rev_pred),
                    gt_ids=tuple(map_rev_true),
                )
            else:
                stats_dict.update(
                    matched_pairs=(),
                    matched_scores=(),
                    matched_tps=(),
                    pred_ids=(),
                    gt_ids=(),
                )
        return stats_dict

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single, thresh))


def wrapper_matching_dataset_lazy(stats_all, thresh, criterion="iou", by_image=False):
    set(
        (
            "fp",
            "tp",
            "fn",
            "precision",
            "recall",
            "accuracy",
            "f1",
            "criterion",
            "thresh",
            "n_true",
            "n_pred",
            "mean_true_score",
            "mean_matched_score",
            "panoptic_quality",
        )
    )

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    single_thresh = True if n_threshs == 1 else False
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i, s in enumerate(stats):
            acc = accumulate[i]
            for item in s.items():
                k, v = item
                if k == "mean_true_score" and not bool(by_image):
                    # convert mean_true_score to "sum_matched_score"
                    acc[k] = acc.setdefault(k, 0) + v * s["n_true"]
                else:
                    try:
                        acc[k] = acc.setdefault(k, 0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr, acc in zip(thresh, accumulate):
        acc["criterion"] = criterion
        acc["thresh"] = thr
        acc["by_image"] = bool(by_image)
        if bool(by_image):
            for k in (
                "precision",
                "recall",
                "accuracy",
                "f1",
                "mean_true_score",
                "mean_matched_score",
                "panoptic_quality",
            ):
                acc[k] /= n_images
        else:
            tp, fp, fn, n_true = acc["tp"], acc["fp"], acc["fn"], acc["n_true"]
            sum_matched_score = acc["mean_true_score"]

            mean_matched_score = _safe_divide(sum_matched_score, tp)
            mean_true_score = _safe_divide(sum_matched_score, n_true)
            panoptic_quality = _safe_divide(sum_matched_score, tp + fp / 2 + fn / 2)

            acc.update(
                precision=precision(tp, fp, fn),
                recall=recall(tp, fp, fn),
                accuracy=accuracy(tp, fp, fn),
                f1=f1(tp, fp, fn),
                mean_true_score=mean_true_score,
                mean_matched_score=mean_matched_score,
                panoptic_quality=panoptic_quality,
            )

    accumulate = tuple(
        namedtuple("DatasetMatching", acc.keys())(*acc.values()) for acc in accumulate
    )
    return accumulate[0] if single_thresh else accumulate
