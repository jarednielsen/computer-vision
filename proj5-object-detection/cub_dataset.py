import cv2
import json
import numpy as np

class CUB_Dataset:

    def __init__(self):
        """ This reads in the metadata and prepares everythings for easy access to the data """

        self._sift = cv2.xfeatures2d.SIFT_create()
        self._meta = json.load( open('cub_metadata.json','r') )

        self._bboxes = {}
        for key in self._meta['bounding_boxes']:
            self._bboxes[int(key)] = [int(x) for x in self._meta['bounding_boxes'][key]]

        ipaths = self._meta['image_paths']
        self._imagepaths = dict([ (int(key),ipaths[key]) for key in ipaths])

        # Grab the image numbers for the training/test splits
        splits = self._meta['train_test_split']
        self._train_image_list = []
        for k in splits['training']:
            self._train_image_list.extend(splits['training'][k])
        self._test_image_list = []
        for k in splits['testing']:
            self._test_image_list.extend(splits['testing'][k])


    def getSIFTfeatures( self, imgnum, bbox_only=False ):
        """ This returns the SIFT keypoints and descriptors for the specified image number.  If the bbox_only parameter is set to True, then the keypoints are filtered down to just those within the object bounding box. """
        img = self.getImage(imgnum)
        f,d = self._sift.detectAndCompute(img, None)

        if bbox_only:
            bx,by,bw,bh = self._bboxes[imgnum]
            npts = len(f)
            inside = np.zeros( (npts,), dtype=bool )
            f_in = []
            for p in range(npts):
                x,y = f[p].pt
                s   = f[p].size
                inside[p] = (x >= bx+s) and (x <= bx+bw-s) and (y >= by+s) and (y <= by+bh-s)
                if inside[p]:
                    f_in.append(f[p])
            d_in = d[inside,:]
            return f_in, d_in
        else:
            return f,d


    def getImage( self, imgnum, bbox_only=False ):
        """ This returns the image (a numpy array) for the specified image number.  If the bbox_only parameter is set to True, then the image returned is only the image of the ojbect itself. """
        img = cv2.imread( 'images/{}'.format(self._imagepaths[imgnum]) )[:,:,::-1]
        if bbox_only:
            x,y,w,h = self._bboxes[imgnum]
            img = img[y:(y+h),x:(x+w),:]
        return img

    def getTrainingSet(self):
        """ Returns a list of the TRAINING set image numbers.  All image numbers in the CUB dataset are between 1 and 11788, inclusive ."""
        return self._train_image_list


    def getTestSet(self):
        """ Returns a list of the TEST set image numbers.  All image numbers in the CUB dataset are between 1 and 11788, inclusive. """
        return self._test_image_list
