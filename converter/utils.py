import numpy as np
import h5py
import SimpleITK as sitk

import glob
import os
import pydicom

import cv2
from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def save_as_nii(data, save_path):
    sitk_data = sitk.GetImageFromArray(data)
    sitk.WriteImage(sitk_data, save_path)


'''
## dicom series reader by simpleITK
def dicom_series_reader(data_path):
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(data_path)
  reader.SetFileNames(dicom_names)
  data = reader.Execute()
  image_array = sitk.GetArrayFromImage(data).astype(np.float32)

  return data,image_array
'''

'''Note
pydicom is faster than simpleITK
e.g. one sample consist of 214 slices
   - pydicom: 1.1s
   - simpleITK: 6.5s    

'''


## dicom series reader by pydicom, rt and series in different folders
def dicom_series_reader(data_path):
    dcms = glob.glob(os.path.join(data_path, '*.dcm'))
    try:
        meta_data = [pydicom.read_file(dcm) for dcm in dcms]
    except:
        meta_data = [pydicom.read_file(dcm,force=True) for dcm in dcms]
        for i in range(len(meta_data)):
            meta_data[i].file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta_data.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    images = np.stack([s.pixel_array for s in meta_data],axis=0).astype(np.float32)
    # pixel value transform to HU
    # images [images == -2000] = 0
    images = images * meta_data[0].RescaleSlope + meta_data[0].RescaleIntercept
    return meta_data, images


## dicom series reader by pydicom
def dicom_series_reader_without_postfix(data_path):
    dcms = glob.glob(os.path.join(data_path, 'CT*'))
    dcms = [dcm for dcm in dcms if "dir" not in dcm]
    try:
        meta_data = [pydicom.read_file(dcm) for dcm in dcms]
    except:
        meta_data = [pydicom.read_file(dcm,force=True) for dcm in dcms]
        for i in range(len(meta_data)):
            meta_data[i].file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    meta_data.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    images = np.stack([s.pixel_array for s in meta_data],axis=0).astype(np.float32)
    # pixel value transform to HU
    # images [images == -2000] = 0
    images = images * meta_data[0].RescaleSlope + meta_data[0].RescaleIntercept
    return meta_data, images


## nii.gz reader
def nii_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return data,image


def trunc_gray(img, in_range=(-1000, 600)):
    img = img - in_range[0]
    scale = in_range[1] - in_range[0]
    img[img < 0] = 0
    img[img > scale] = scale

    return img
    

def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img


def get_roi(sample, keep_size=24, pad_flag=False):

    image = sample['image']
    label = sample['label']

    h,w = image.shape
    roi = get_body(image)

    if np.sum(roi) != 0:
        roi_nz = np.nonzero(roi)
        roi_bbox = [
            np.maximum((np.amin(roi_nz[0]) - self.keep_size), 0), # left_top x
            np.maximum((np.amin(roi_nz[1]) - self.keep_size), 0), # left_top y
            np.minimum((np.amax(roi_nz[0]) + self.keep_size), h), # right_bottom x
            np.minimum((np.amax(roi_nz[1]) + self.keep_size), w)  # right_bottom y
        ]
    else:
        roi_bbox = [0,0,h,w]

    image = image[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
    label = label[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
    # pad
    if pad_flag:
        nh, nw = roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]
        if abs(nh - nw) > 1:
            if nh > nw:
                pad = ((0,0),(int(nh-nw)//2,int(nh-nw)//2))
            else:
                pad = ((int(nw-nh)//2,int(nw-nh)//2),(0,0))
            image = np.pad(image,pad,'constant')
            label = np.pad(label,pad,'constant')

    sample['image'] = image
    sample['label'] = label
    sample['bbox'] = roi_bbox
    return sample

def get_body_2d(image):
    body_array = np.zeros_like(image, dtype=np.uint8)
    img = rescale_intensity(image, out_range=(0, 255))
    img = img.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body = cv2.erode(img, kernel, iterations=1)
    blur = cv2.GaussianBlur(body, (5, 5), 0)
    _, body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area = [[c, cv2.contourArea(contours[c])] for c in range(len(contours))]
    area.sort(key=lambda x: x[1], reverse=True)
    body = np.zeros_like(body, dtype=np.uint8)
    for j in range(min(len(area),3)):
        if area[j][1] > area[0][1] / 20:
            contour = contours[area[j][0]]
            r = contour[:, 0, 1]
            c = contour[:, 0, 0]
            rr, cc = polygon(r, c)
            body[rr, cc] = 1
    body_array = cv2.medianBlur(body, 5)

    return body_array


if __name__ == "__main__":
    data_path = '/staff/shijun/dataset/Nasopharynx_Oar/200301_CT'
    meta_data,image = dicom_series_reader(data_path)
    print(image.shape)
    print(meta_data[0].SliceThickness)