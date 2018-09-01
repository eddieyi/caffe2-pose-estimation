
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import glob
import logging
import os
import sys
import math
import matplotlib
import pylab as plt
import time
import numpy as np
import logging
import contextlib
import cPickle as pickle
from six import string_types
from collections import OrderedDict

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, scope

def initialize_gpu_from_weights_file(model, weights_file, gpu_id=0):
    """Initialize a network with ops on a specific GPU. If you use CUDA_VISIBLE_DEVICES to target specific GPUs, 
    Caffe2 will automatically map logical GPU ids (starting from 0) to the physical GPUs specified in CUDA_VISIBLE_DEVICES.
    """
    logger = logging.getLogger(__name__)

    logger.info('Loading weights from: {}'.format(weights_file))
    ws_blobs = workspace.Blobs()
    with open(weights_file, 'r') as f:
        src_blobs = pickle.load(f)
    if 'cfg' in src_blobs:
        print('------------cfg exist-----------------------------')
    if 'blobs' in src_blobs:
        # Backwards compat--dictionary used to be only blobs, now they are stored under the 'blobs' key
        src_blobs = src_blobs['blobs']
    # Initialize weights on GPU gpu_id only
    unscoped_param_names = OrderedDict()  # Print these out in model order
    for blob in model.params:
        unscoped_param_names[UnscopeName(str(blob))] = True
    with NamedCudaScope(gpu_id):
        for unscoped_param_name in unscoped_param_names.keys():
            if (unscoped_param_name.find(']_') >= 0 and unscoped_param_name not in src_blobs):
                # Special case for sharing initialization from a pretrained model: If a blob named '_[xyz]_foo' is in model.params and not
                # in the initialization blob dictionary, then load source blob 'foo' into destination blob '_[xyz]_foo'
                src_name = unscoped_param_name[
                    unscoped_param_name.find(']_') + 2:]
            else:
                src_name = unscoped_param_name
            if src_name not in src_blobs:
                logger.info('{:s} not found'.format(src_name))
                continue
            dst_name = core.ScopedName(unscoped_param_name)
            has_momentum = src_name + '_momentum' in src_blobs
            has_momentum_str = ' [+ momentum]' if has_momentum else ''
            logger.debug( '{:s}{:} loaded from weights file into {:s}: {}'.format( src_name, has_momentum_str, dst_name, src_blobs[src_name].shape ) )
            if dst_name in ws_blobs:
                # If the blob is already in the workspace, make sure that it matches the shape of the loaded blob
                ws_blob = workspace.FetchBlob(dst_name)
                assert ws_blob.shape == src_blobs[src_name].shape, ('Workspace blob {} with shape {} does not match weights file shape {}').format(
                        src_name, ws_blob.shape, src_blobs[src_name].shape)
            workspace.FeedBlob( dst_name, src_blobs[src_name].astype(np.float32, copy=False) )
            if has_momentum:
                workspace.FeedBlob( dst_name + '_momentum', src_blobs[src_name + '_momentum'].astype(np.float32, copy=False) )

    # We preserve blobs that are in the weights file but not used by the current model. We load these into CPU memory under the '__preserve__/' namescope.
    # These blobs will be stored when saving a model to a weights file. This feature allows for alternating optimization of Faster R-CNN in which blobs
    # unused by one step can still be preserved forward and used to initialize another step.
    for src_name in src_blobs.keys():
        if (src_name not in unscoped_param_names and
                not src_name.endswith('_momentum') and
                src_blobs[src_name] is not None):
            with CpuScope():
                workspace.FeedBlob('__preserve__/{:s}'.format(src_name), src_blobs[src_name])
                logger.debug('{:s} preserved in workspace (unused)'.format(src_name))

def UnscopeName(possibly_scoped_name):
    """Remove any name scoping from a (possibly) scoped name. For example,
    convert the name 'gpu_0/foo' to 'foo'."""
    assert isinstance(possibly_scoped_name, string_types)
    return possibly_scoped_name[possibly_scoped_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]

@contextlib.contextmanager
def CpuScope():
    """Create a CPU device scope."""
    cpu_dev = core.DeviceOption(caffe2_pb2.CPU)
    with core.DeviceScope(cpu_dev):
        yield

@contextlib.contextmanager
def NamedCudaScope(gpu_id):
    with core.NameScope('gpu_{:d}'.format(gpu_id)):
        gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
        with core.DeviceScope(gpu_dev):
            yield

def joint_grouping(heatmaps_limb, heatmaps_joint, imgHeight):
    logger = logging.getLogger(__name__)

    import scipy
    from scipy.ndimage.filters import gaussian_filter

    thre1 = 0.1
    thre2 = 0.05
    mid_num = 10

    all_peaks = []
    peak_counter = 0
    for part in range(heatmaps_joint.shape[2] - 1):
        x_list = []
        y_list = []
        map_ori = heatmaps_joint[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > thre1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks] # [(x1,y1,score1),(x2,y2,score2),...]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))] # [(x1,y1,score1,0),(x2,y2,score2,1),...]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    connection_all = []
    special_k = []
    mid_num = 10

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], [1,16], [16,18], [3,17], [6,18]]
    # the middle joints heatmap correspondence
    mapIdx=[[31,32],[39,40],[33,34],[35,36],[41,42],[43,44],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[47,48],[49,50],[53,54],[51,52],[55,56],[37,38],[45,46]]
    
    for k in range(len(mapIdx)):
        score_mid = heatmaps_limb[:,:,[x-19 for x in mapIdx[k]]]
        indexA, indexB = limbSeq[k]
        candA = all_peaks[indexA - 1] # [(x1,y1,score1,0),(x2,y2,score2,1),...]
        candB = all_peaks[indexB - 1]
        nA = len(candA)
        nB = len(candB)
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    if norm > 1e-6:
                        vec = np.divide(vec, norm)
                        
                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*imgHeight/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
                    else:
                        connection_candidate.append([i, j, 0, candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist]) # all_peaks: [(x1,y1,score1,0),(x2,y2,score2,1),...]

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return candidate, subset

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def im_infer_keypoints(model, imgFolderDir, blob_input, blob_output_limb, blob_output_joint, imgSavePath, scales=[1], target_height=368, \
                       mergeFlag='HM_AVG', image_ext='jpg', visFlag=True, netStride=8, gpu_id=0):
    """Computes keypoint predictions with test-time augmentations. """
    logger = logging.getLogger(__name__)

    max_size = 3000000 # the size of the scaled image cannot exceed this number, can be updated according to the GPU memory size

    assert os.path.isdir(imgFolderDir), 'The provided path not exist or is not a directory !'
    imgpath_list = glob.iglob(imgFolderDir + '/*.' + image_ext)
    processSumTime = 0.0
    processCnt = 0
    for i, imgpath in enumerate(imgpath_list):    
        oriImg = cv2.imread(imgpath) # B, G, R order, in np.uint8 type
        assert oriImg is not None, 'Failed to read image \'{}\''.format(imgpath)
        height, width = oriImg.shape[:2]
        if oriImg.shape == (height, width):
            logger.debug('Convert 1-channel image !')
            oriImg = np.stack((oriImg,)*3, -1)
        
        pixel_means = np.zeros([1,1,3], dtype=np.float32)
        pixel_means[0,0,0] = 128.0
        pixel_means[0,0,1] = 128.0
        pixel_means[0,0,2] = 128.0
        im = oriImg - pixel_means # normalize pixel values to [-128, 128]

        pixel_scale = np.zeros([1,1,3], dtype=np.float32)
        pixel_scale[0,0,0] = 255.0
        pixel_scale[0,0,1] = 255.0
        pixel_scale[0,0,2] = 255.0
        im /= pixel_scale # scale pixel values to [-0.5, 0.5]

        pred_heatmaps_limb_set = [] # collect heatmaps predicted under different scales
        pred_heatmaps_joint_set = []
        multiplier = [x * target_height / oriImg.shape[0] for x in scales]
        for m in range(len(multiplier)): # Compute detections at different scales
            scale = multiplier[m]
            imageToTest = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = padRightDownCorner(imageToTest, netStride, padValue=0)
            heightBeforePad = imageToTest_padded.shape[0] - pad[2]
            widthBeforePad  = imageToTest_padded.shape[1] - pad[3]
            
            imageToTest_padded = imageToTest_padded.astype(np.float32)
            imageToTest_padded = imageToTest_padded[np.newaxis, :, :, :] # In case of 1, or if im.ndim == 3: im = np.expand_dims(im, axis=0)
            channel_swap = (0, 3, 1, 2) # Move channels (axis 3) to axis 1, axis order will become: (batch elem, channel, height, width)
            imageToTest_padded = imageToTest_padded.transpose(channel_swap)

            t = time.time()
            workspace.FeedBlob('gpu_{}/{}'.format(gpu_id, blob_input), imageToTest_padded)
            workspace.RunNet(model.net.Proto().name)
            delta_t = time.time() - t
            processSumTime += delta_t
            processCnt += 1
            
            pred_heatmaps_limb = np.squeeze( workspace.FetchBlob(core.ScopedName(blob_output_limb)), axis=0 )
            pred_heatmaps_limb = pred_heatmaps_limb.transpose([1, 2, 0])
            pred_heatmaps_limb = cv2.resize(pred_heatmaps_limb, None, None, fx=netStride, fy=netStride, interpolation=cv2.INTER_CUBIC)
            pred_heatmaps_limb = pred_heatmaps_limb[:heightBeforePad, :widthBeforePad, :]
            pred_heatmaps_limb = cv2.resize(pred_heatmaps_limb, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            pred_heatmaps_joint = np.squeeze( workspace.FetchBlob(core.ScopedName(blob_output_joint)), axis=0 )
            pred_heatmaps_joint = pred_heatmaps_joint.transpose([1, 2, 0])
            pred_heatmaps_joint = cv2.resize(pred_heatmaps_joint, None, None, fx=netStride, fy=netStride, interpolation=cv2.INTER_CUBIC)
            pred_heatmaps_joint = pred_heatmaps_joint[:heightBeforePad, :widthBeforePad, :]
            pred_heatmaps_joint = cv2.resize(pred_heatmaps_joint, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            pred_heatmaps_limb_set.append(pred_heatmaps_limb)
            pred_heatmaps_joint_set.append(pred_heatmaps_joint)
        
        if i == 0:
            logger.info('Note: inference on the first image will be slower than the rest (caches and auto-tuning need to warm up)')
        
        if mergeFlag == 'HM_AVG':
            pred_heatmaps_limb_merge = np.mean(pred_heatmaps_limb_set, axis=0) # combining the heatmaps
            pred_heatmaps_joint_merge = np.mean(pred_heatmaps_joint_set, axis=0)
        elif mergeFlag == 'HM_MAX':
            pred_heatmaps_limb_merge = np.amax(pred_heatmaps_limb_set, axis=0)
            pred_heatmaps_joint_merge = np.amax(pred_heatmaps_joint_set, axis=0)
        else:
            raise NotImplementedError('Heuristic {} not supported'.format(mergeFlag))

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18],[3,17],[6,18]]

        candidate, subset = joint_grouping(pred_heatmaps_limb_merge, pred_heatmaps_joint_merge, height)

        if visFlag == True:
            colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], \
                      [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
            stickwidth = 4
            
            canvas = oriImg.copy() # B,G,R order
            cur_canvas = canvas.copy()
            for kk1 in range(17): # only plot 17 connections
                for n in range(len(subset)):
                    index = subset[n][np.array(limbSeq[kk1])-1]
                    if -1 in index:
                        continue
                    Y = candidate[index.astype(int), 0]
                    X = candidate[index.astype(int), 1]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[kk1])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            plt.figure()
            plt.imshow(canvas[:,:,[2,1,0]])
            plt.axis('off')
            plt.title('Predictions', fontweight="bold", size=10)
            # fig = matplotlib.pyplot.gcf()
            # fig.set_size_inches(12, 12)
            # imgname = os.path.basename(imgpath)
            # plt.savefig('%s/%s' %(imgSavePath, imgname), bbox_inches='tight')

            plt.show()

    logger.info('The total inference time: {:.3f}s, total process count: {}, average inference time: {:.3f}s'.format(processSumTime, processCnt, processSumTime / float(processCnt) ) )
