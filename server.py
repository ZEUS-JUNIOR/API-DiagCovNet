
from PIL import Image
from io import BytesIO, FileIO
import numpy as np
import tensorflow as tf
import pydicom as dicom
import tensorflow
import segmentation_models as sm
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
import nibabel as nib
import SimpleITK as sitk

"""      """


"""
def read_image_all(path):
    if path.endswith("dcm"):
        image = dicom.dcmread(path)
        image = np.array(image.pixel_array)
        image = image.reshape((256, 256))
    elif path.endswith("png") or path.endswith("jpg"):
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (256, 256))
    else:
        image = None
    return image
"""

input_shape = [256, 256]


def read_image_numpy(path):
    if path.endswith("npy"):
        image = np.load(path)
    else:
        image = None
    return image


"""
def normalize_function(img, air, fat):
    air_HU = -1000
    fat_HU = -100

    delta_air_fat_HU = abs(air_HU - fat_HU)
    delta_air = abs(air - air_HU)
    delta_fat_air_rgb = abs(fat - air)
    ratio = delta_air_fat_HU / delta_fat_air_rgb

    img = img - air
    img = img * ratio
    img = img + air_HU
    return img
"""


def write_image(filepath, image_array):
    image = filepath.replace('.dcm', '.jpg')
    cv2.imwrite(filepath, image_array)


def preprocess_images(images_arr, mean_std=None):
    images_arr[images_arr > 500] = 500
    images_arr[images_arr < -1500] = -1500
    min_perc, max_perc = np.percentile(images_arr, 5), np.percentile(images_arr, 95)
    images_arr_valid = images_arr[(images_arr > min_perc) & (images_arr < max_perc)]
    mean, std = (images_arr_valid.mean(), images_arr_valid.std()) if mean_std is None else mean_std
    images_arr = (images_arr - mean) / std
    print(f' taille ::::  {mean}')
    return images_arr, (mean, std)


def visualize(image_batch, mask_batch=None, pred_batch=None, num_samples=8):
    num_classes = mask_batch.shape[-1] if mask_batch is not None else 0
    fix, ax = plt.subplots(num_classes + 1, num_samples, figsize=(num_samples * 2, (num_classes + 1) * 2))

    for i in range(num_samples):
        ax_image = ax[0, i] if num_classes > 0 else ax[i]
        ax_image.imshow(image_batch[i, :, :, 0], cmap='Greys')
        ax_image.set_xticks([])
        ax_image.set_yticks([])

        if mask_batch is not None:
            for j in range(num_classes):
                if pred_batch is None:
                    mask_to_show = mask_batch[i, :, :, j]
                else:
                    mask_to_show = np.zeros(shape=(*mask_batch.shape[1:-1], 3))
                    mask_to_show[..., 0] = pred_batch[i, :, :, j] > 0.5
                    mask_to_show[..., 1] = mask_batch[i, :, :, j]
                ax[j + 1, i].imshow(mask_to_show, vmin=0, vmax=1)
                ax[j + 1, i].set_xticks([])
                ax[j + 1, i].set_yticks([])

    plt.tight_layout()
    plt.show()


def fscore_glass(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., 0:1],
                               y_pred[..., 0:1])


def fscore_consolidation(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., 1:2],
                               y_pred[..., 1:2])


def fscore_lungs_other(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., 2:3],
                               y_pred[..., 2:3])


def fscore_glass_and_consolidation(y_true, y_pred):
    return sm.metrics.f1_score(y_true[..., :2],
                               y_pred[..., :2])


model = tensorflow.keras.models.load_model('_models/best_model',
                                           compile=False,
                                           custom_objects={
                                               'categorical_crossentropy': sm.losses.categorical_crossentropy,
                                               'fscore_consolidation': fscore_consolidation,
                                               'fscore_glass': fscore_glass,
                                               'fscore_lungs_other': fscore_lungs_other,
                                               'fscore_glass_and_consolidation': fscore_glass_and_consolidation})
model.compile(Adam(lr=0.001, amsgrad=True),
              loss=sm.losses.jaccard_loss)




def read_image(image_encoded):
    print(type(BytesIO(image_encoded)))
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = np.array(ct_scan.get_fdata())
    array = resize(array, (256, 256, 3), preserve_range=True)
    resized_img = array[:, :, 0]
    fl_resized_img = np.fliplr(np.rot90(resized_img, k=3))
    return fl_resized_img


def read_image_seg(image_encoded):
    '''
    Reads .nii file and returns pixel array
    '''
    new_nifti = np.zeros((512, 512, 0))
    numpyImage = Image.open(BytesIO(image_encoded))
    numpyImage = np.asfarray(numpyImage).astype("float64")  # à supprimer
    # image=cv2.resize(numpyImage, (512, 512))
    image = resize(numpyImage, (256, 256, 3), preserve_range=True)
    resized_img = image[:, :, 0]
    fl_resized_img = np.fliplr(np.rot90(resized_img, k=3))

    # normalized_slice = normalize_function(fl_resized_img, 0, -800)
    return fl_resized_img


""" lecture pour un chemin fuilepath
def read_image_seg(filepath, i=0):
    '''
    Reads .nii file and returns pixel array
    '''
    inputImage =sitk.ReadImage(filepath)
    numpyImage = sitk.GetArrayFromImage(inputImage)
    numpyImage = np.array(numpyImage).astype("float64")
    # image=cv2.resize(numpyImage, (512, 512))
    image = resize(numpyImage, (256, 256, 3), preserve_range=True)
    resized_img = image[:, :, 0]
    fl_resized_img = np.fliplr(np.rot90(resized_img, k=3))
    normalized_slice = normalize_function(fl_resized_img, 0, -1000, -1027.6803)
    return normalized_slice
"""


def read_dicom(filepath):
    dcmfile = dicom.dcmread(filepath)
    dcm_numpy = dcmfile.pixel_array
    return dcm_numpy


def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    image = np.asfarray(image)
    return image


def load_model_():
    model = model = tf.keras.models.load_model('12.h5')
    return model


def load_model_seg():
    model = tensorflow.keras.models.load_model('_models/best_model',
                                               compile=False,
                                               custom_objects={
                                                   'categorical_crossentropy': sm.losses.categorical_crossentropy,
                                                   'fscore_consolidation': fscore_consolidation,
                                                   'fscore_glass': fscore_glass,
                                                   'fscore_lungs_other': fscore_lungs_other,
                                                   'fscore_glass_and_consolidation': fscore_glass_and_consolidation})
    model.compile(Adam(lr=0.001, amsgrad=True),
                  loss=sm.losses.jaccard_loss)
    return model


def predict(image: np.ndarray):
    _model = load_model_()
    ide = _model.predict(image.reshape(1, 256, 256, 1), batch_size=1)
    re = np.argmax(ide, axis=1)[0]
    return re


def predict_segmentation(pathfile):
    # model = load_model_seg()

    img = read_image_numpy('image test set/test_img1.dcm.npy')
    img, _ = preprocess_images(img, mean_std=(-451.6488342285156, 458.5679016113281))
    preds = model.predict(np.expand_dims(img, axis=0))
    plt.imshow(preds[0, :, :, 0])
    fix, ax = plt.subplots(5, 1, sharex=True)
    for i in range(5):
        if i == 0:
            ax[i].imshow(img.reshape(512, 512), cmap='Greys')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        else:
            ax[i].imshow(preds[0, :, :, i - 1])
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.savefig("1.png")
    img = cv2.imread("1.png")
    return img


"""
def predict_segmentation(image: np.ndarray):
    # model = load_model_seg()

    preds, _ = preprocess_images(image, mean_std=(-451.6488342285156, 458.5679016113281))
    preds = model.predict(np.expand_dims(preds, axis=0))
    fix, ax = plt.subplots(5, 1, sharex=True)
    for i in range(5):
        if i == 0:
            ax[i].imshow(image)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        else:
            ax[i].imshow(preds[0, :, :, i-1], cmap='jet')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.savefig("1.png")
    img = cv2.imread("1.png")
    return img
"""



if __name__ == '__main__':
    img = predict_segmentation('image test set/test_img1.dcm.npy')
    plt.imshow(img)
    plt.show(block=True)
    """
    model = load_model_seg()
    test_preds = model.predict(img.reshape(1, 256, 256, 1), batch_size=1)
    #visualize(test_preds, mask_batch=None, pred_batch=None, num_samples=1)
    fix, ax = plt.subplots(4, 1, sharex=True, figsize=(1, (2) * 2))
    for i in range(4):
        ax[i].imshow(test_preds[0, :, :, i])
    plt.show(block=False)
    #cv2.imshow("image", test_preds)
    #test_masks_prediction = test_preds > 0.5
    #print(test_preds)
    print(110001)
    """
