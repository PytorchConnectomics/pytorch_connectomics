import numpy as np
from .augmentor import DataAugment

from scipy.ndimage.interpolation import map_coordinates, zoom
import numbers
from skimage.draw import line
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation

class MissingParts(DataAugment):
    """Missing-parts augmentation of image stacks.

    Args:
        deformation_strength (int): Default: 0
        iterations (int): Default: 40
        deform_ratio (float): Default: 0.25
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, 
                 deformation_strength=0, 
                 iterations=40, 
                 deform_ratio=0.25, 
                 p=0.5):
        super(MissingParts, self).__init__(p=p)
        self.deformation_strength = deformation_strength
        self.iterations = iterations
        self.set_params()

    def set_params(self):
        # No change in sample size
        pass

    def prepare_deform_slice(self, slice_shape, random_state):
        # grow slice shape by 2 x deformation strength
        grow_by = 2 * self.deformation_strength
        #print ('sliceshape: '+str(slice_shape[0])+' growby: '+str(grow_by)+ ' strength: '+str(deformation_strength))
        shape = (slice_shape[0] + grow_by, slice_shape[1] + grow_by)
        # randomly choose fixed x or fixed y with p = 1/2
        fixed_x = random_state.rand() < 0.5
        if fixed_x:
            x0, y0 = 0, np.random.randint(1, shape[1] - 2)
            x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
        else:
            x0, y0 = np.random.randint(1, shape[0] - 2), 0
            x1, y1 = np.random.randint(1, shape[0] - 2), shape[1] - 1

        ## generate the mask of the line that should be blacked out
        #print (shape)
        line_mask = np.zeros(shape, dtype='bool')
        rr, cc = line(x0, y0, x1, y1)
        line_mask[rr, cc] = 1

        # generate vectorfield pointing towards the line to compress the image
        # first we get the unit vector representing the line
        line_vector = np.array([x1 - x0, y1 - y0], dtype='float32')
        line_vector /= np.linalg.norm(line_vector)
        # next, we generate the normal to the line
        normal_vector = np.zeros_like(line_vector)
        normal_vector[0] = - line_vector[1]
        normal_vector[1] = line_vector[0]

        # make meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # generate the vector field
        flow_x, flow_y = np.zeros(shape), np.zeros(shape)

        # find the 2 components where coordinates are bigger / smaller than the line
        # to apply normal vector in the correct direction
        components, n_components = label(np.logical_not(line_mask).view('uint8'))
        assert n_components == 2, "%i" % n_components
        neg_val = components[0, 0] if fixed_x else components[-1, -1]
        pos_val = components[-1, -1] if fixed_x else components[0, 0]

        flow_x[components == pos_val] = self.deformation_strength * normal_vector[1]
        flow_y[components == pos_val] = self.deformation_strength * normal_vector[0]
        flow_x[components == neg_val] = - self.deformation_strength * normal_vector[1]
        flow_y[components == neg_val] = - self.deformation_strength * normal_vector[0]

        # generate the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)

        # dilate the line mask
        line_mask = binary_dilation(line_mask, iterations=self.iterations) #default=10
        
        return flow_x, flow_y, line_mask

    def deform_2d(self, image2d, random_state):
        flow_x, flow_y, line_mask = self.prepare_deform_slice(image2d.shape, random_state)
        section = image2d.squeeze()
        mean = section.mean()
        shape = section.shape
        #interpolation=3
        section = map_coordinates(section, (flow_y, flow_x), mode='constant', 
                        order=3).reshape(int(flow_x.shape[0]**0.5),int(flow_x.shape[0]**0.5))
        section = np.clip(section, 0., 1.)
        section[line_mask] = mean
        return section 

    def apply_deform(self, imgs, random_state):
        transformedimgs = np.copy(imgs)
        sectionsnum = imgs.shape[0]
        i=0
        while i < sectionsnum:
            if random_state.rand() < self.p:
                transformedimgs[i] = self.deform_2d(imgs[i], random_state)
                i += 2 # only one deformed image in any consecutive 3 images
            i += 1
        return transformedimgs

    def __call__(self, data, random_state=np.random):
        augmented = self.apply_deform(data['image'], random_state)
        data['image'] = augmented
        return data
