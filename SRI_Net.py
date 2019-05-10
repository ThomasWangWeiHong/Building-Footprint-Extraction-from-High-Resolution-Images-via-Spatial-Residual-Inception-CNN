import cv2
import glob
import json
import numpy as np
import rasterio
from group_norm import GroupNormalization
from keras import backend as K
from keras import regularizers
from keras.models import Input, Model
from keras.layers import Activation, concatenate, Conv2D, MaxPooling2D, UpSampling2D, SeparableConv2D
from keras.optimizers import Adam



def training_mask_generation(input_image_filename, input_geojson_filename):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        image = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        
    mask = np.zeros((image.shape[0], image.shape[1]))
    
    ulx = metadata['transform'][2]
    xres = metadata['transform'][0]
    uly = metadata['transform'][5]
    yres = metadata['transform'][4]
                                      
    lrx = ulx + (image.shape[1] * xres)                                                         
    lry = uly - (image.shape[0] * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((image.shape[1]) ** 2 / (image.shape[1] + 1)) / (lrx - ulx)
        yf = ((image.shape[0]) ** 2 / (image.shape[0] + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask, position, 1)
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, image_height_size, image_width_size, mode, percentage_overlap, 
                                      buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image and its mask in the form of fixed size segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((image_array.shape[0], x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = mask_array
        
    img_list = []
    mask_list = []
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, 0]
            mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
            mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
            mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
            mask_flip_hor = cv2.flip(mask_original, 0)
            mask_flip_vert = cv2.flip(mask_original, 1)
            mask_flip_both = cv2.flip(mask_original, -1)
            mask_list.extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, mask_flip_vert, 
                              mask_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(mask_list), image_height_size, image_width_size, 1))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        mask_segment_array[index, :, :, 0] = mask_list[index]
        
    return image_segment_array, mask_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\' + 'Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Training Polygons' + '\\Train_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    
    for file in range(len(img_files)):
        with rasterio.open(img_files[file]) as f:
            metadata = f.profile
            img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
            
        mask = training_mask_generation(img_files[file], polygon_files[file])
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 0, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 1, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 2, 
                                                                      percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 3, 
                                                                      percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    
    return img_full_array, mask_full_array



def dice_coef(y_true, y_pred):
    """ 
    This function generates the dice coefficient for use in semantic segmentation model training. 
    
    """
    
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    coef = (2 * intersection) / (K.sum(y_true_flat) + K.sum(y_pred_flat))
    
    return coef



def dice_coef_loss(y_true, y_pred):
    """ 
    This function generates the dice coefficient loss function for use in semantic segmentation model training. 
    
    """
    
    return -dice_coef(y_true, y_pred)



def sri_net(img_height_size, img_width_size, n_bands, group_filters, kernel_reg_decay_rate = 0.0001, initial_filters = 64, 
            base_depth_1 = 64, feat_map_filters = 256, l_r = 0.0001, decay_rate = 0.9):
    """
    This function is used to generate the Spatial Residual Inception (SRI) architecture as described in the paper 'Building 
    Footprint Extraction From High - Resolution Images via Spatial Residual Inception Convolutional Neural Network' by Liu P., 
    Liu X., Liu M., Shi Q., Yang J., Xu X., Zhang Y. (2019).
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - group_filters: Number of groups to be used for group normalization
    - kernel_reg_decay_rate: L2 regularizer decay rate to be used for convolution kernels
    - initial_filters: Number of filters to be applied to input image
    - base_depth_1: Base depth of bottleneck 1
    - feat_map_filters: Number of filters to be left over after filtering the initial feature maps
    - l_r: Learning rate to be applied for the Adam optimizer
    - decay_rate: Learning decay rate for the Adam optimizer
    
    Outputs:
    - sri_net_model: SRI - Net model to be trained using input parameters and network architecture
    
    """
    
    if (initial_filters % group_filters != 0) or (base_depth_1 % group_filters != 0):
        raise ValueError('Please make sure that initial filters and base_depth_1 are multiples of group_filters.')
        
    if (feat_map_filters % 2 != 0):
        raise ValueError('Please make sure that feat_map_filters is an even number.')
        
    base_depth_2 = int(2 * base_depth_1)
    base_depth_3 = int(2 * base_depth_2)
    base_depth_4 = int(2 * base_depth_3)
        
    
    
    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    
    
    
    input_conv = Conv2D(initial_filters, (7, 7), strides = (2, 2), padding = 'same', 
                        kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                        bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(img_input)
    input_conv_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(input_conv)
    input_conv_act = Activation('relu')(input_conv_gn)
    
    
    
    botneck_1_input = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(input_conv_act)
    
    
    
    botneck_1_i1_input_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_input)
    botneck_1_i1_input_act = Activation('relu')(botneck_1_i1_input_gn)
    botneck_1_i1_c1 = Conv2D(base_depth_1, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_i1_input_act)
    botneck_1_i1_c1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_i1_c1)
    botneck_1_i1_c1_act = Activation('relu')(botneck_1_i1_c1_gn)
    botneck_1_i1_c2 = SeparableConv2D(base_depth_1, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_i1_c1_act)
    botneck_1_i1_c2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_i1_c2)
    botneck_1_i1_c2_act = Activation('relu')(botneck_1_i1_c2_gn)
    botneck_1_i1_c3 = Conv2D(base_depth_1, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_i1_c2_act)
    botneck_1_i1_c3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_i1_c3)
    f2 = concatenate([botneck_1_i1_input_act, botneck_1_i1_c3_gn])
    
    
    botneck_1_c1_input_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(f2)
    botneck_1_c1_input_act = Activation('relu')(botneck_1_c1_input_gn)
    botneck_1_c1_c1 = Conv2D(base_depth_1, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_c1_input_act)
    botneck_1_c1_c1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_c1_c1)
    botneck_1_c1_c1_act = Activation('relu')(botneck_1_c1_c1_gn)
    botneck_1_c1_c2 = SeparableConv2D(base_depth_1, (5, 5), strides = (2, 2), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_c1_c1_act)
    botneck_1_c1_c2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_c1_c2)
    botneck_1_c1_c2_act = Activation('relu')(botneck_1_c1_c2_gn)
    botneck_1_c1_c3 = Conv2D(base_depth_1, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_c1_c2_act)
    botneck_1_c1_c3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(botneck_1_c1_c3)
    down_1 = Conv2D(base_depth_1, (1, 1), strides = (2, 2), padding = 'same', 
                    kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                    bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_1_c1_input_act)
    down_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(down_1)
    botneck_1_out = concatenate([botneck_1_c1_c3_gn, down_1_gn])
    
    
    
    botneck_2_i1_input_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_1_out)
    botneck_2_i1_input_act = Activation('relu')(botneck_2_i1_input_gn)
    botneck_2_i1_c1 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_i1_input_act)
    botneck_2_i1_c1_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i1_c1)
    botneck_2_i1_c1_act = Activation('relu')(botneck_2_i1_c1_gn)
    botneck_2_i1_c2 = SeparableConv2D(base_depth_2, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_i1_c1_act)
    botneck_2_i1_c2_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i1_c2)
    botneck_2_i1_c2_act = Activation('relu')(botneck_2_i1_c2_gn)
    botneck_2_i1_c3 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_i1_c2_act)
    botneck_2_i1_c3_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i1_c3)
    botneck_2_i1_out = concatenate([botneck_2_i1_input_act, botneck_2_i1_c3_gn])
    
    
    botneck_2_i2_input_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i1_out)
    botneck_2_i2_input_act = Activation('relu')(botneck_2_i2_input_gn)
    botneck_2_i2_c1 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_i2_input_act)
    botneck_2_i2_c1_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i2_c1)
    botneck_2_i2_c1_act = Activation('relu')(botneck_2_i2_c1_gn)
    botneck_2_i2_c2 = SeparableConv2D(base_depth_2, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_i2_c1_act)
    botneck_2_i2_c2_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i2_c2)
    botneck_2_i2_c2_act = Activation('relu')(botneck_2_i2_c2_gn)
    botneck_2_i2_c3 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_i2_c2_act)
    botneck_2_i2_c3_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_i2_c3)
    f3 = concatenate([botneck_2_i2_input_act, botneck_2_i2_c3_gn])
    
    
    botneck_2_c1_input_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(f3)
    botneck_2_c1_input_act = Activation('relu')(botneck_2_c1_input_gn)
    botneck_2_c1_c1 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_c1_input_act)
    botneck_2_c1_c1_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_c1_c1)
    botneck_2_c1_c1_act = Activation('relu')(botneck_2_c1_c1_gn)
    botneck_2_c1_c2 = SeparableConv2D(base_depth_2, (5, 5), strides = (2, 2), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_c1_c1_act)
    botneck_2_c1_c2_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_c1_c2)
    botneck_2_c1_c2_act = Activation('relu')(botneck_2_c1_c2_gn)
    botneck_2_c1_c3 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_c1_c2_act)
    botneck_2_c1_c3_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_c1_c3)
    down_2 = Conv2D(base_depth_2, (1, 1), strides = (2, 2), padding = 'same', 
                    kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                    bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_2_c1_input_act)
    down_2_gn = GroupNormalization(groups = int(2 * group_filters), axis = -1, epsilon = 0.1)(down_2)
    botneck_2_out = concatenate([botneck_2_c1_c3_gn, down_2_gn])
    
    
    
    botneck_3_i1_input_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_2_out)
    botneck_3_i1_input_act = Activation('relu')(botneck_3_i1_input_gn)
    botneck_3_i1_c1 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i1_input_act)
    botneck_3_i1_c1_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i1_c1)
    botneck_3_i1_c1_act = Activation('relu')(botneck_3_i1_c1_gn)
    botneck_3_i1_c2 = SeparableConv2D(base_depth_3, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i1_c1_act)
    botneck_3_i1_c2_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i1_c2)
    botneck_3_i1_c2_act = Activation('relu')(botneck_3_i1_c2_gn)
    botneck_3_i1_c3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i1_c2_act)
    botneck_3_i1_c3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i1_c3)
    botneck_3_i1_out = concatenate([botneck_3_i1_input_act, botneck_3_i1_c3_gn])
    
    
    botneck_3_i2_input_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i1_out)
    botneck_3_i2_input_act = Activation('relu')(botneck_3_i2_input_gn)
    botneck_3_i2_c1 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i2_input_act)
    botneck_3_i2_c1_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i2_c1)
    botneck_3_i2_c1_act = Activation('relu')(botneck_3_i2_c1_gn)
    botneck_3_i2_c2 = SeparableConv2D(base_depth_3, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i2_c1_act)
    botneck_3_i2_c2_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i2_c2)
    botneck_3_i2_c2_act = Activation('relu')(botneck_3_i2_c2_gn)
    botneck_3_i2_c3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i2_c2_act)
    botneck_3_i2_c3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i2_c3)
    botneck_3_i2_out = concatenate([botneck_3_i2_input_act, botneck_3_i2_c3_gn])
    
    
    botneck_3_i3_input_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i2_out)
    botneck_3_i3_input_act = Activation('relu')(botneck_3_i3_input_gn)
    botneck_3_i3_c1 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i3_input_act)
    botneck_3_i3_c1_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i3_c1)
    botneck_3_i3_c1_act = Activation('relu')(botneck_3_i3_c1_gn)
    botneck_3_i3_c2 = SeparableConv2D(base_depth_3, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i3_c1_act)
    botneck_3_i3_c2_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i3_c2)
    botneck_3_i3_c2_act = Activation('relu')(botneck_3_i3_c2_gn)
    botneck_3_i3_c3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i3_c2_act)
    botneck_3_i3_c3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i3_c3)
    botneck_3_i3_out = concatenate([botneck_3_i3_input_act, botneck_3_i3_c3_gn])
    
    
    botneck_3_i4_input_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i3_out)
    botneck_3_i4_input_act = Activation('relu')(botneck_3_i4_input_gn)
    botneck_3_i4_c1 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i4_input_act)
    botneck_3_i4_c1_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i4_c1)
    botneck_3_i4_c1_act = Activation('relu')(botneck_3_i4_c1_gn)
    botneck_3_i4_c2 = SeparableConv2D(base_depth_3, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i4_c1_act)
    botneck_3_i4_c2_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i4_c2)
    botneck_3_i4_c2_act = Activation('relu')(botneck_3_i4_c2_gn)
    botneck_3_i4_c3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i4_c2_act)
    botneck_3_i4_c3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i4_c3)
    botneck_3_i4_out = concatenate([botneck_3_i4_input_act, botneck_3_i4_c3_gn])
    
    
    botneck_3_i5_input_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i4_out)
    botneck_3_i5_input_act = Activation('relu')(botneck_3_i5_input_gn)
    botneck_3_i5_c1 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i5_input_act)
    botneck_3_i5_c1_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i5_c1)
    botneck_3_i5_c1_act = Activation('relu')(botneck_3_i5_c1_gn)
    botneck_3_i5_c2 = SeparableConv2D(base_depth_3, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i5_c1_act)
    botneck_3_i5_c2_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i5_c2)
    botneck_3_i5_c2_act = Activation('relu')(botneck_3_i5_c2_gn)
    botneck_3_i5_c3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_i5_c2_act)
    botneck_3_i5_c3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i5_c3)
    botneck_3_i5_out = concatenate([botneck_3_i5_input_act, botneck_3_i5_c3_gn])
    
    
    botneck_3_c1_input_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_i5_out)
    botneck_3_c1_input_act = Activation('relu')(botneck_3_c1_input_gn)
    botneck_3_c1_c1 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_c1_input_act)
    botneck_3_c1_c1_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_c1_c1)
    botneck_3_c1_c1_act = Activation('relu')(botneck_3_c1_c1_gn)
    botneck_3_c1_c2 = SeparableConv2D(base_depth_3, (5, 5), padding = 'same', dilation_rate = (2, 2),
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_c1_c1_act)
    botneck_3_c1_c2_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_c1_c2)
    botneck_3_c1_c2_act = Activation('relu')(botneck_3_c1_c2_gn)
    botneck_3_c1_c3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_c1_c2_act)
    botneck_3_c1_c3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_c1_c3)
    conv_3 = Conv2D(base_depth_3, (1, 1), padding = 'same', 
                    kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                    bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_3_c1_input_act)
    conv_3_gn = GroupNormalization(groups = int(4 * group_filters), axis = -1, epsilon = 0.1)(conv_3)
    botneck_3_out = concatenate([botneck_3_c1_c3_gn, conv_3_gn])
    
    
    
    botneck_4_i1_input_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_3_out)
    botneck_4_i1_input_act = Activation('relu')(botneck_4_i1_input_gn)
    botneck_4_i1_c1 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i1_input_act)
    botneck_4_i1_c1_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i1_c1)
    botneck_4_i1_c1_act = Activation('relu')(botneck_4_i1_c1_gn)
    botneck_4_i1_c2 = SeparableConv2D(base_depth_4, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i1_c1_act)
    botneck_4_i1_c2_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i1_c2)
    botneck_4_i1_c2_act = Activation('relu')(botneck_4_i1_c2_gn)
    botneck_4_i1_c3 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i1_c2_act)
    botneck_4_i1_c3_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i1_c3)
    botneck_4_i1_out = concatenate([botneck_4_i1_input_act, botneck_4_i1_c3_gn])
    
    
    botneck_4_i2_input_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i1_out)
    botneck_4_i2_input_act = Activation('relu')(botneck_4_i2_input_gn)
    botneck_4_i2_c1 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i2_input_act)
    botneck_4_i2_c1_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i2_c1)
    botneck_4_i2_c1_act = Activation('relu')(botneck_4_i2_c1_gn)
    botneck_4_i2_c2 = SeparableConv2D(base_depth_4, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i2_c1_act)
    botneck_4_i2_c2_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i2_c2)
    botneck_4_i2_c2_act = Activation('relu')(botneck_4_i2_c2_gn)
    botneck_4_i2_c3 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i2_c2_act)
    botneck_4_i2_c3_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i2_c3)
    botneck_4_i2_out = concatenate([botneck_4_i2_input_act, botneck_4_i2_c3_gn])
    
    
    botneck_4_i3_input_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i2_out)
    botneck_4_i3_input_act = Activation('relu')(botneck_4_i3_input_gn)
    botneck_4_i3_c1 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i3_input_act)
    botneck_4_i3_c1_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i3_c1)
    botneck_4_i3_c1_act = Activation('relu')(botneck_4_i3_c1_gn)
    botneck_4_i3_c2 = SeparableConv2D(base_depth_4, (5, 5), padding = 'same', 
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i3_c1_act)
    botneck_4_i3_c2_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i3_c2)
    botneck_4_i3_c2_act = Activation('relu')(botneck_4_i3_c2_gn)
    botneck_4_i3_c3 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_i3_c2_act)
    botneck_4_i3_c3_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i3_c3)
    botneck_4_i3_out = concatenate([botneck_4_i3_input_act, botneck_4_i3_c3_gn])
    
    
    botneck_4_c1_input_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_i3_out)
    botneck_4_c1_input_act = Activation('relu')(botneck_4_c1_input_gn)
    botneck_4_c1_c1 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_c1_input_act)
    botneck_4_c1_c1_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_c1_c1)
    botneck_4_c1_c1_act = Activation('relu')(botneck_4_c1_c1_gn)
    botneck_4_c1_c2 = SeparableConv2D(base_depth_4, (5, 5), padding = 'same', dilation_rate = (2, 2),
                                      depthwise_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                      pointwise_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_c1_c1_act)
    botneck_4_c1_c2_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_c1_c2)
    botneck_4_c1_c2_act = Activation('relu')(botneck_4_c1_c2_gn)
    botneck_4_c1_c3 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                             kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                             bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_c1_c2_act)
    botneck_4_c1_c3_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(botneck_4_c1_c3)
    conv_4 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                    kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                    bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(botneck_4_c1_input_act)
    conv_4_gn = GroupNormalization(groups = int(8 * group_filters), axis = -1, epsilon = 0.1)(conv_4)
    f4 = concatenate([botneck_4_c1_c3_gn, conv_4_gn])
    
    
    
    sri_lv_4_input = Activation('relu')(f4)
    sri_lv_4_initial_filter = Conv2D(int(feat_map_filters / 2), (1, 1), padding = 'same', 
                                     kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                     bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_input)
    sri_lv_4_incep_1 = Conv2D(base_depth_4, (1, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_initial_filter)
    sri_lv_4_incep_2_1 = Conv2D(base_depth_4, (1, 3), padding = 'same', 
                                kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_initial_filter)
    sri_lv_4_incep_2_2 = Conv2D(base_depth_4, (3, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_incep_2_1)
    sri_lv_4_incep_3_1 = Conv2D(base_depth_4, (1, 7), padding = 'same', 
                                kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_initial_filter)
    sri_lv_4_incep_3_2 = Conv2D(base_depth_4, (7, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_incep_3_1)
    sri_lv_4_incep_out = concatenate([sri_lv_4_incep_1, sri_lv_4_incep_2_2, sri_lv_4_incep_3_2])
    sri_lv_4_incep_filter = Conv2D(int(feat_map_filters / 2), (1, 1), padding = 'same', 
                                   kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                   bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_4_incep_out)
    sri_lv_4_out = concatenate([sri_lv_4_initial_filter, sri_lv_4_incep_filter])
    sri_lv_4_out_act = Activation('relu')(sri_lv_4_out)
    
    
    
    f4_sri_upsam = UpSampling2D(interpolation = 'bilinear')(sri_lv_4_out_act)
    f3_filter = Conv2D(feat_map_filters, (1, 1), padding = 'same', 
                       kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                       bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(f3)
    f3_combined = concatenate([f4_sri_upsam, f3_filter])
    
    
    
    sri_lv_3_input = Activation('relu')(f3_combined)
    sri_lv_3_initial_filter = Conv2D(int(feat_map_filters / 2), (1, 1), padding = 'same', 
                                     kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                     bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_input)
    sri_lv_3_incep_1 = Conv2D(base_depth_2, (1, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_initial_filter)
    sri_lv_3_incep_2_1 = Conv2D(base_depth_2, (1, 3), padding = 'same', 
                                kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_initial_filter)
    sri_lv_3_incep_2_2 = Conv2D(base_depth_2, (3, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_incep_2_1)
    sri_lv_3_incep_3_1 = Conv2D(base_depth_2, (1, 7), padding = 'same', 
                                kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_initial_filter)
    sri_lv_3_incep_3_2 = Conv2D(base_depth_2, (7, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_incep_3_1)
    sri_lv_3_incep_out = concatenate([sri_lv_3_incep_1, sri_lv_3_incep_2_2, sri_lv_3_incep_3_2])
    sri_lv_3_incep_filter = Conv2D(int(feat_map_filters / 2), (1, 1), padding = 'same', 
                                   kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                   bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_3_incep_out)
    sri_lv_3_out = concatenate([sri_lv_3_initial_filter, sri_lv_3_incep_filter])
    sri_lv_3_out_act = Activation('relu')(sri_lv_3_out)
    
    
    
    f3_sri_upsam = UpSampling2D(interpolation = 'bilinear')(sri_lv_3_out_act)
    f2_filter = Conv2D(feat_map_filters, (1, 1), padding = 'same', 
                       kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                       bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(f2)
    f2_combined = concatenate([f3_sri_upsam, f2_filter])
    
    
    
    sri_lv_2_input = Activation('relu')(f2_combined)
    sri_lv_2_initial_filter = Conv2D(int(feat_map_filters / 2), (1, 1), padding = 'same', 
                                     kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                     bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_input)
    sri_lv_2_incep_1 = Conv2D(base_depth_1, (1, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_initial_filter)
    sri_lv_2_incep_2_1 = Conv2D(base_depth_1, (1, 3), padding = 'same', 
                                kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_initial_filter)
    sri_lv_2_incep_2_2 = Conv2D(base_depth_1, (3, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_incep_2_1)
    sri_lv_2_incep_3_1 = Conv2D(base_depth_1, (1, 7), padding = 'same', 
                                kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_initial_filter)
    sri_lv_2_incep_3_2 = Conv2D(base_depth_1, (7, 1), padding = 'same', 
                              kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                              bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_incep_3_1)
    sri_lv_2_incep_out = concatenate([sri_lv_2_incep_1, sri_lv_2_incep_2_2, sri_lv_2_incep_3_2])
    sri_lv_2_incep_filter = Conv2D(int(feat_map_filters / 2), (1, 1), padding = 'same', 
                                   kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                                   bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(sri_lv_2_incep_out)
    sri_lv_2_out = concatenate([sri_lv_2_initial_filter, sri_lv_2_incep_filter])
    sri_lv_2_out_act = Activation('relu')(sri_lv_2_out)
    
    
    
    f2_sri_upsam = UpSampling2D(size = (4, 4), interpolation = 'bilinear')(sri_lv_2_out_act)
    final_filter = Conv2D(feat_map_filters, (3, 3), padding = 'same', 
                          kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                          bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(f2_sri_upsam)
    
    
    
    pred_layer = Conv2D(1, (3, 3), padding = 'same', activation = 'sigmoid',
                        kernel_regularizer = regularizers.l2(kernel_reg_decay_rate), 
                        bias_regularizer = regularizers.l2(kernel_reg_decay_rate))(final_filter)
    
    sri_net_model = Model(inputs = img_input, outputs = pred_layer)
    sri_net_model.compile(loss = dice_coef_loss, optimizer = Adam(lr = l_r, decay = decay_rate), metrics = [dice_coef])
    
    return sri_net_model



def image_model_predict(input_image_filename, output_filename, img_height_size, img_width_size, fitted_model, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for prediction. The 
    output mask is then allocated to its corresponding location in the image in order to obtain the complete mask for the 
    entire image without being constrained by image size. 
    
    Inputs:
    - input_image_filename: File path of image file for which prediction is to be conducted
    - output_filename: File path of output predicted binary raster mask file
    - img_height_size: Height of image patches to be used for model prediction
    - img_height_size: Width of image patches to be used for model prediction
    - fitted_model: Trained keras model which is to be used for prediction
    - write: Boolean indicating whether to write predicted binary raster mask to file
    
    Output:
    - mask_complete: Numpy array of predicted binary raster mask for input image
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
     
    y_size = ((img.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((img.shape[1] // img_width_size) + 1) * img_width_size
    
    if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, img.shape[1], img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((img.shape[0], x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    else:
         img_complete = img
            
    mask = np.zeros((img_complete.shape[0], img_complete.shape[1], 1))
    img_holder = np.zeros((1, img_height_size, img_width_size, img.shape[2]))
    
    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img.shape[2]]
            preds = fitted_model.predict(img_holder)
            mask[i : i + img_height_size, j : j + img_width_size, 0] = preds[0, :, :, 0]
            
    mask_complete = np.expand_dims(mask[0 : img.shape[0], 0 : img.shape[1], 0], axis = 2)
    mask_complete = np.transpose(mask_complete, [2, 0, 1]).astype('float32')
    
    
    if write:
        metadata['count'] = 1
        metadata['dtype'] = 'float32'
        
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(mask_complete)
    
    return mask_complete


