import numpy as np
from .augmentor import DataAugment

class MissingParts(DataAugment):
    """Missing-parts augmentation of image stacks
    Args:
        deformation_strength (int):
        iterations (int): (default: 80)
        deform_ratio (float): 
        p (float): probability of applying the augmentation.
    """
    def __init__(self, 
                 deformation_strength=0, 
                 iterations=80, 
                 deform_ratio=0.25, 
                 p=0.5):
        super(MissingParts, self).__init__(p=p)
        self.displacement = 16

    def set_params(self):
        # No change in sample size
        pass

    def prepare_deform_slice(slice_shape, deformation_strength, iterations):
        # grow slice shape by 2 x deformation strength
        grow_by = 2 * deformation_strength
        #print ('sliceshape: '+str(slice_shape[0])+' growby: '+str(grow_by)+ ' strength: '+str(deformation_strength))
        shape = (slice_shape[0] + grow_by, slice_shape[1] + grow_by)
        # randomly choose fixed x or fixed y with p = 1/2
        fixed_x = np.random.random() < 0.5
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

        flow_x[components == pos_val] = deformation_strength * normal_vector[1]
        flow_y[components == pos_val] = deformation_strength * normal_vector[0]
        flow_x[components == neg_val] = - deformation_strength * normal_vector[1]
        flow_y[components == neg_val] = - deformation_strength * normal_vector[0]

        # generate the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)

        # dilate the line mask
        line_mask = binary_dilation(line_mask, iterations=iterations)#default=10
        
        return flow_x, flow_y, line_mask

    def deform_2d(image2d, deformation_strength, iterations):
        flow_x, flow_y, line_mask = prepare_deform_slice(image2d.shape,deformation_strength,iterations)
        section = image2d.squeeze()
        mean = section.mean()
        shape = section.shape
        #interpolation=3
        section = map_coordinates(section, (flow_y, flow_x), mode='constant', 
                                order=3).reshape(int(flow_x.shape[0]**0.5),int(flow_x.shape[0]**0.5))
        section = np.clip(section, 0., 1.)
        section[line_mask] = mean
        return section 

    def apply_deform(imgs, deformation_strength=0, iterations=80, deform_ratio=0.25):
        '''
        imgs :3D
        '''
        transformedimgs = np.copy(imgs)
        sectionsnum = imgs.shape[0]
        i =0
        while i < sectionsnum:
            if random.random() <= deform_ratio:
                transformedimgs[i] = deform_2d(imgs[i], deformation_strength, iterations)
                i += 2
            i += 1
        return transformedimgs