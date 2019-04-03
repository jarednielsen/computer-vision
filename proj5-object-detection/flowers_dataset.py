import cv2
import json
import numpy as np

class Flowers_Dataset:

    def __init__(self):
        """ This reads in the metadata and prepares everythings for easy access to the data """

        self._sift = cv2.xfeatures2d.SIFT_create()
        self._meta = json.load( open('flowers_metadata.json','r') )

        ipaths = self._meta['image_paths']
        self._imagepaths = dict([ (int(key),ipaths[key]) for key in ipaths])

        # Grab the image numbers for the training/test splits
        splits = self._meta['splits']
        self._train_image_list = splits['training']
        self._test_image_list = splits['testing']
        self._test_image_list.extend(splits['validation'])


    def getSIFTfeatures( self, imgnum ):
        """ This returns the SIFT keypoints and descriptors for the specified image number.  """
        img = self.getImage(imgnum)
        f,d = self._sift.detectAndCompute(img, None)
        return f,d


    def getImage( self, imgnum ):
        """ This returns the image (a numpy array) for the specified image number. """
        #print( '{}'.format(self._imagepaths[imgnum]) )
        img = cv2.imread( '{}'.format(self._imagepaths[imgnum]) )[:,:,::-1]
        return img

    def getTrainingSet(self):
        """ Returns a list of the TRAINING set image numbers.  All image numbers in the FLOWERS dataset are between 1 and 8189, inclusive ."""
        return self._train_image_list


    def getTestSet(self):
        """ Returns a list of the TEST set image numbers.  All image numbers in the FLOWERS dataset are between 1 and 8189, inclusive. """
        return self._test_image_list
