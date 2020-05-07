import argparse
import numpy as np
from scipy import ndimage
import h5py

class Clefts:

    def __init__(self, test, truth):

        test_clefts = test
        truth_clefts = truth

        self.resolution=(40.0, 4.0, 4.0)

        self.test_clefts_mask = (test_clefts == 0).astype(int)
        self.truth_clefts_mask = (truth_clefts == 0).astype(int)
    
        self.test_clefts_edt = ndimage.distance_transform_edt(self.test_clefts_mask, sampling=self.resolution)
        self.truth_clefts_edt = ndimage.distance_transform_edt(self.truth_clefts_mask, sampling=self.resolution)


    def count_false_positives(self, threshold = 200):

        mask1 = np.invert(self.test_clefts_mask)
        mask2 = self.truth_clefts_edt > threshold
        false_positives = self.truth_clefts_edt[np.logical_and(mask1, mask2)]
        return false_positives.size

    def count_false_negatives(self, threshold = 200):

        mask1 = np.invert(self.truth_clefts_mask)
        mask2 = self.test_clefts_edt > threshold
        false_negatives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        return false_negatives.size

    def acc_false_positives(self):

        mask = np.invert(self.test_clefts_mask)
        ADGT = (self.truth_clefts_edt * mask).sum() / mask.sum()
        return ADGT

    def acc_false_negatives(self):

        mask = np.invert(self.truth_clefts_mask)
        ADF = (self.test_clefts_edt * mask).sum() / mask.sum()
        return ADF



def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-p','--prediction',  type=str, help='prediction path')
    parser.add_argument('-g','--groundtruth', type=str, help='groundtruth path')
    args = parser.parse_args()
    return args                    

def main():
    args = get_args()

    print('0. load data')
    test = h5py.File(name=args.prediction, mode='r',  libver='latest')['main']
    test = np.array(test)
    test[test < 128] = 0
    test = (test != 0).astype(np.uint8)

    truth = h5py.File(name=args.groundtruth, mode='r',  libver='latest')['main']
    truth = np.array(truth)
    truth = (truth != 0).astype(np.uint8)

    assert (test.shape == truth.shape)
    print('Test volume shape:', test.shape)
    total_pixels = np.prod(test.shape)
    print('Total number of pixels:', total_pixels)

    print('1. Start evaluation:')
    clefts_evaluation = Clefts(test, truth)
    false_positive_count = clefts_evaluation.count_false_positives()
    false_negative_count = clefts_evaluation.count_false_negatives()
    false_positive_rate = false_positive_count/total_pixels
    false_negative_rate = false_negative_count/total_pixels
    true_positive_rate = 1 - false_negative_rate
    f1_score = 2*true_positive_rate/(2*true_positive_rate+false_positive_rate+false_negative_rate)
    ADGT = clefts_evaluation.acc_false_positives()
    ADF = clefts_evaluation.acc_false_negatives()
    CRIME_score = (ADGT+ADF)/2

    print('\tFalse positive rate: %.4f' %(false_positive_rate))
    print('\tFalse negative rate: %.4f' %(false_negative_rate))
    print('\tF1 score: %.4f' %(f1_score))
    print('\tADGT: %.4f' %(ADGT))
    print('\tADF: %.4f' %(ADF))
    print('\tCRIME_score: %.4f' %(CRIME_score))
    print('2. End evaluation.')

if __name__ == "__main__":
    main()    