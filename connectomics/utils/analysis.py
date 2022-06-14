import h5py  # read h5 files
import numpy as np  # handle volume data
import pandas as pd  # use pandas data frames for plotting with seaborn
from tqdm import tqdm  # show progress
# derive the center of mass of instances
from scipy.ndimage.measurements import center_of_mass


def voxel_instance_size_h5(h5_target: str, ds_name: str) -> pd.DataFrame:
    # Loads the data from an h5 file. See -> voxel_instance_size()
    return voxel_instance_size(np.asarray(read_h5py(h5_target)), ds_name)


def distance_nn(h5_target: str, ds_name: str, iso=[1, 1, 1]) -> pd.DataFrame:
    # Loads the data from an h5 file. See -> distance_nn()
    return distance_nn(np.asarray(read_h5py(h5_target)), ds_name, iso=iso)


def pixel_intensity(h5_source: str, h5_target: str, bOrF: str, ds_name: str) -> pd.DataFrame:
    # Loads the data from the h5 files. See -> pixel_intensity()
    return pixel_intensity(np.asarray(read_h5py(h5_source)), np.asarray(read_h5py(h5_target)), bOrF, ds_name)


def voxel_instance_size(target: np.ndarray, ds_name: str) -> pd.DataFrame:
    ''' Calculate the voxel based size of each instance in an instance segmentation map.

        Args:
            target: The target data as numpy ndarray
            ds_name: Name of the dataset, saved in pd column

        Return
            A single column Panda data frame that contains the voxel based instance sizes.
    '''

    # find the unique values and their number of accurancy
    idx, count = np.unique(target, return_counts=True)

    # save the pixel count to a pandas data frame
    idx_pix_count = {x: y for x, y in zip(
        idx[1:], count[1:])}  # 1:, skip background
    idx_pix_count_pd = pd.DataFrame(data=list(idx_pix_count.values()),
                                    columns=["Size"], index=list(idx_pix_count.keys()))

    # add column with dataset name
    idx_pix_count_pd["Dataset"] = ds_name

    return idx_pix_count_pd


def distance_nn(target: np.ndarray, ds_name: str, iso=[1, 1, 1]) -> pd.DataFrame:
    ''' Caculate the distance to the NN for each instance in the target matrix. 

        Args:
            target: The target data as numpy ndarray
            iso: Axis scaling factor in case of anisotropy
            ds_name: Name of the dataset, saved in pd column

        Return
            A single column Panda data frame that contains the distance of each instance to its NN
    '''

    # convert the instance map to binary
    binary = (target != 0).astype(np.uint8)

    # derive the center of mass of each instance in the target matrix
    cm = center_of_mass(binary, target, list(np.unique(target))[1:])

    distance = []

    # calculate for each instance the distance to the NN
    for i, vi in tqdm(enumerate(list(cm))):
        # the second argument of closest_node() is a list with the coordinates of
        # all instances center of mass, except of the cm of the current selected instane
        distance.append(closest_node(vi, np.array(
            cm)[np.arange(len(list(cm))) != i]))

    # write the distance value to a pandas data frame
    idx_zxy_values_pd = pd.DataFrame(data=list(distance),
                                     columns=["NN_Distance"])

    # add column with dataset name
    idx_zxy_values_pd["Dataset"] = ds_name

    return idx_zxy_values_pd


def closest_node(point, points):
    ''' Calculate the distance between a point and a list of points.
        Returns the shortest distance. Used by: distance_nn()

        Args:
            point: A single point with x,y or x,y,z values
            points: A list of points with x,y or x,y,z values

        Return
    '''
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.min(dist_2)


def pixel_intensity(source: np.ndarray, target: np.ndarray, bOrF: str, ds_name: str) -> pd.DataFrame:
    ''' Retrives the intesity of each pixel. Writes them to a Pandas data frame.
        Can handle background for foreground.
        Args:
            source: Source numpy ndarray
            target: Target numpy ndarray
            bOrF: Either 'Foreground' or 'Background', indicates which intensities to estimate
            ds_name: Name of the dataset, saved in pd column

        Return
            A pandas frame with the intensity of each pixel, if the pixel belongs to the background
            or foreground, and the dataset it belongs to.
    '''

    # mask out forground or background
    assert (bOrF == 'Foreground' or bOrF ==
            'Background'), f"bOrF has to equal \"Foreground\" or \"Background\", not {bOrF}"
    mask_bOrF = 1 if bOrF == 'Foreground' else 0

    # create mask with all values greater 0
    mask = target > 0

    # mask out forground
    masked_source = source[mask == mask_bOrF]

    # create the pd data frame
    pix_int_count_front_pd = pi_pd(masked_source, bOrF, ds_name)

    return pix_int_count_front_pd


def pi_pd(mask: np.ndarray, bOrF: str, ds_name: str) -> pd.DataFrame:
    ''' Creates pandas data frame of the intesity of all pixels in the mask.
        Used by: pixel_intensity()

        Args:
            mask: Numpy array of masked out pixels 
            bOrF: Either 'Forground' or 'Background', indicates which intesities are estimated
            ds_name: Name of the dataset

        Return
            Pandas data frame of the intesities of each pixel in the mask array

    '''
    # convert masked matrix to 1D array and write to pd
    mask_1D = mask.ravel()
    pix_int_count_pd = pd.DataFrame(data=mask_1D, columns=["Intensity"])

    # add column with back or forground specification
    pix_int_count_pd["B/F"] = bOrF

    # add column with dataset name
    pix_int_count_pd["Dataset"] = ds_name

    return pix_int_count_pd


def read_h5py(file_name: str, key: str = 'main') -> list:
    ''' Read h5 file.

        Args:
            file_name: Path to the h5 file
            key: Key under which the data is stored in the h5 file

        Return
            The data as list.
    '''
    data = None
    with h5py.File(file_name, 'r') as h5:
        data = list(h5.get(key))
    return data