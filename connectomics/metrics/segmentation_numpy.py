from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential

from connectomics.utils.label_overlap import compute_label_overlap

matching_criteria = dict()

__all__ = [
    "adapted_rand",
    "voi",
    "instance_matching",
    "instance_matching_simple",
    "matching_criteria",
]


def adapted_rand_oracle(seg, gt, gt_ids=None):
    """Efficiently compute per-GT-segment oracle ARE by incremental update.

    For each GT segment, computes the ARE that would result from perfectly
    predicting that segment (``pred[gt == g_id] = unique_id``) while
    keeping everything else unchanged.

    Builds the contingency table once, then for each GT segment updates
    only the affected row — O(nnz_in_row) per GT instead of O(volume).

    Parameters
    ----------
    seg : np.ndarray
        Predicted segmentation.
    gt : np.ndarray, same shape as seg
        Ground-truth segmentation.
    gt_ids : array-like, optional
        GT segment IDs to evaluate.  If None, uses all non-zero GT IDs.

    Returns
    -------
    list of (gt_id, are_oracle, delta_are)
        Sorted by delta_are descending (most impactful first).
        ``delta_are = are_baseline - are_oracle`` (positive = improvement).
    """
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = int(np.amax(segA)) + 1
    n_labels_B = int(np.amax(segB)) + 1

    # Build contingency table: p_ij[gt_label, pred_label] = count
    ones_data = np.ones(n, int)
    p_ij = sparse.csr_matrix(
        (ones_data, (segA, segB)), shape=(n_labels_A, n_labels_B)
    )

    # Baseline quantities (exclude GT background row 0)
    a = p_ij[1:n_labels_A, :]           # all pred cols including bg
    b = p_ij[1:n_labels_A, 1:n_labels_B]  # exclude pred bg col
    c = np.asarray(p_ij[1:n_labels_A, 0].todense()).ravel()  # pred bg col

    a_i = np.asarray(a.sum(1)).ravel()   # GT row sums
    b_i = np.asarray(b.sum(0)).ravel()   # pred col sums (excl bg)

    sum_c = np.sum(c)
    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + sum_c / n
    sumAB = np.sum(b.multiply(b)) + sum_c / n

    prec_base = sumAB / sumB if sumB > 0 else 0
    rec_base = sumAB / sumA if sumA > 0 else 0
    f_base = 2.0 * prec_base * rec_base / (prec_base + rec_base) if (prec_base + rec_base) > 0 else 0
    are_base = 1.0 - f_base

    if gt_ids is None:
        gt_ids = np.arange(1, n_labels_A)
    else:
        gt_ids = np.asarray(gt_ids)

    # New pred label ID (beyond any existing)
    new_label = n_labels_B  # 0-indexed in the b matrix → col index = new_label - 1

    results = []
    for g_id in gt_ids:
        g_id = int(g_id)
        if g_id < 1 or g_id >= n_labels_A:
            continue
        row_idx = g_id - 1  # index into a/b/c (which start from GT label 1)

        # Current row entries in b (pred labels excl bg)
        row = b.getrow(row_idx)
        row_data = row.data.copy()  # nonzero values
        row_cols = row.indices.copy()  # column indices

        g_size = int(a_i[row_idx])  # total voxels in this GT segment
        if g_size == 0:
            continue

        # Old contributions from this row to sumAB and sumB
        old_pij_sq = np.sum(row_data * row_data)
        old_c_val = c[row_idx]

        # Old b_i contributions for affected pred columns
        old_bi_affected = b_i[row_cols].copy()

        # --- After oracle fix: row becomes [0, ..., 0, g_size] at new column ---
        # sumA is unchanged (row sum = g_size, same as before)

        # sumAB change:
        #   remove old: sum(p_ij^2) for this row + old_c/n
        #   add new: g_size^2 (single entry) + 0/n (no bg overlap)
        new_sumAB = sumAB - old_pij_sq - old_c_val / n + g_size * g_size

        # sumB change:
        #   affected old columns: b_i[col] decreases by row_data[j]
        #   new column: b_i[new] = g_size
        #   remove old_c contribution, add 0 (oracle has no bg)
        new_sumB = sumB - old_c_val / n
        # Update affected columns
        for j in range(len(row_cols)):
            col = row_cols[j]
            val = row_data[j]
            old_sq = old_bi_affected[j] ** 2
            new_bi = old_bi_affected[j] - val
            new_sumB = new_sumB - old_sq + new_bi * new_bi
        # Add new column
        new_sumB = new_sumB + g_size * g_size

        prec = new_sumAB / new_sumB if new_sumB > 0 else 0
        rec = new_sumAB / sumA if sumA > 0 else 0
        f = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        are_oracle = 1.0 - f
        delta = are_base - are_oracle

        results.append((g_id, are_oracle, delta))

    results.sort(key=lambda x: -x[2])
    return results, are_base


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


def voi(reconstruction, groundtruth, ignore_reconstruction=None, ignore_groundtruth=None):
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
    if ignore_reconstruction is None:
        ignore_reconstruction = []
    if ignore_groundtruth is None:
        ignore_groundtruth = [0]
    hyxg, hxgy = split_vi(reconstruction, groundtruth, ignore_reconstruction, ignore_groundtruth)
    return (hxgy, hyxg)


def split_vi(x, y=None, ignore_x=None, ignore_y=None):
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
    if ignore_x is None:
        ignore_x = [0]
    if ignore_y is None:
        ignore_y = [0]
    _, _, _, hxgy, hygx, _, _ = vi_tables(x, y, ignore_x, ignore_y)
    # false merges, false splits
    return np.array([hygx.sum(), hxgy.sum()])


def vi_tables(x, y=None, ignore_x=None, ignore_y=None):
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
    if ignore_x is None:
        ignore_x = [0]
    if ignore_y is None:
        ignore_y = [0]
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


def contingency_table(seg, gt, ignore_seg=None, ignore_gt=None, norm=True):
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
    if ignore_seg is None:
        ignore_seg = [0]
    if ignore_gt is None:
        ignore_gt = [0]
    segr = seg.ravel()
    gtr = gt.ravel()
    data = np.ones(len(gtr))
    ignored = np.isin(segr, ignore_seg) | np.isin(gtr, ignore_gt)
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
        if isinstance(out, sparse.csr_matrix):
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
    return compute_label_overlap(x, y)


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
    """Calculate detection/instance segmentation metrics between ground truth and predictions.

    Currently, the following metrics are implemented:
        'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1',
        'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score',
        'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp),
    false positives (fp), and false negatives (fn) when their intersection over
    union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives
    * mean_true_score is the mean IoUs of matched true positives but normalized
      by the total number of GT objects
    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation",
      CVPR 2019

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
        if True, additionally calculate matched_pairs and matched_scores
        (returns gt-pred pairs even when scores are below 'thresh')

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
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0,
             recall=0, accuracy=0, f1=0, n_true=1, n_pred=1,
             mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

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
    if not (0 <= np.min(scores) <= np.max(scores) <= 1):
        raise ValueError(
            f"Scores must be in [0, 1], got range [{np.min(scores)}, {np.max(scores)}]"
        )

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


def instance_matching_simple(y_true, y_pred, thresh=0.5, criterion="iou"):
    """Calculate relaxed instance segmentation metrics without Hungarian matching.

    WARNING: This is a RELAXED metric for debugging/analysis only, NOT for benchmark ranking.
    Unlike instance_matching(), this does NOT use optimal bipartite matching (Hungarian algorithm).
    Instead, it simply counts all (GT, Pred) pairs with IoU >= threshold as true positives.

    This metric is useful for:
    - Quick debugging and sanity checks
    - Understanding raw overlap statistics
    - Comparing with strict Hungarian-based metrics

    Metrics computed:
        'tp', 'fp', 'fn', 'precision', 'recall', 'accuracy', 'f1',
        'criterion', 'thresh', 'n_true', 'n_pred'

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default 'iou')

    Returns
    -------
    Dictionary with metrics (tp, fp, fn, precision, recall, accuracy, f1, etc.)

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true, 5, axis=0)
    >>> stats = instance_matching_simple(y_true, y_pred)
    >>> print(f"Accuracy: {stats['accuracy']:.3f}")
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

    thresh = float(thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    if not (0 <= np.min(scores) <= np.max(scores) <= 1):
        raise ValueError(
            f"Scores must be in [0, 1], got range [{np.min(scores)}, {np.max(scores)}]"
        )

    # ignoring background
    scores = scores[1:, 1:]
    n_true, n_pred = scores.shape

    # Simple counting: any pair with IoU >= thresh counts as TP
    # No Hungarian matching - just count all pairs above threshold
    tp = np.sum(scores >= thresh)
    fp = n_pred - tp
    fn = n_true - tp

    stats_dict = dict(
        criterion=criterion,
        thresh=thresh,
        fp=fp,
        tp=tp,
        fn=fn,
        precision=precision(tp, fp, fn),
        recall=recall(tp, fp, fn),
        accuracy=accuracy(tp, fp, fn),
        f1=f1(tp, fp, fn),
        n_true=n_true,
        n_pred=n_pred,
    )

    return stats_dict
