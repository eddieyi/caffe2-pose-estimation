#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import math
import importlib
import matplotlib
import pylab as plt

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, cnn, memonger

from models.cmu_coco import cmu_coco
from utils.utils import initialize_gpu_from_weights_file, im_infer_keypoints, NamedCudaScope

# OpenCL may be enabled by default in OpenCV3; disable it because it's not thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Manually clear root loggers to prevent any module that may have called logging.basicConfig() from blocking our logging setup
logging.root.handlers = []
FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger("pose")
logger.setLevel(logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--weights', dest='weights', help='weight file (./models/model_weights.pkl)', default='./models/cmu_coco_iter440000.pkl', type=str)
    parser.add_argument('--image-ext', dest='image_ext', help='image file name extension (default: jpg)', default='jpg', type=str)
    parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id (default: 0)', default=0, type=int)
    parser.add_argument('--imgSavePath', dest='imgSavePath', help='output path (default: ./images/outputs)', default='./images/outputs', type=str)
    parser.add_argument('--imgFolderDir', dest='imgFolderDir', help='image path (default: ./images)', default='./images', type=str)
    parser.add_argument('--target_height', dest='target_height', help='image height (default: 368)', default=368, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def initialize_model(weights_file, networkDefinition, gpu_id=0, MEMONGER_flag=True, blobs_to_keep=[]):
    """Initialize a model. Loads test-time weights and creates the networks in the Caffe2 workspace. """
    model = model_build(networkDefinition)
    if MEMONGER_flag:
        optimize_memory(model, blobs_to_keep=blobs_to_keep)
    initialize_gpu_from_weights_file( model, weights_file, gpu_id=gpu_id )
    add_inference_inputs(model)
    workspace.CreateNet(model.net)
    return model

def add_inference_inputs(model):
    """Create network input blobs used for inference."""

    def create_input_blobs_for_net(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    create_input_blobs_for_net(model.net.Proto())

def model_build(networkDefinition, gpu_id=0):
    model = cnn.CNNModelHelper(name="pose", use_cudnn=True, cudnn_exhaustive_search=False)
    model.target_gpu_id = gpu_id
    return build_generic_detection_model(model, networkDefinition)

def build_generic_detection_model(model, networkDefinition): # model_builder.py
    def _single_gpu_build_func(model):
        blob_out_limb, blob_out_joint = networkDefinition(model)

    with NamedCudaScope(model.target_gpu_id):
        _single_gpu_build_func(model)

    return model

def optimize_memory(model, gpu_id=0, blobs_to_keep=[]):
    """Save GPU memory through releasing blobs."""
    blobs_to_keep_all_devices = set()

    if blobs_to_keep is not None:
        for blob_name in blobs_to_keep:
            blobs_to_keep_all_devices.add("gpu_{}/{}".format(gpu_id, blob_name))
    model.net._net = memonger.release_blobs_when_used(model.net.Proto(), blobs_to_keep_all_devices)

def main(args):
    # model definition
    netDefinition = cmu_coco

    if netDefinition == cmu_coco:
        blob_input = 'data'
        blob_output_limb = 'Mconv7_stage6_L1'
        blob_output_joint = 'Mconv7_stage6_L2'
        blobs_to_keep = ['conv4_4_CPM', 'concat_stage2', 'concat_stage3', 'concat_stage4', 'concat_stage5', 'concat_stage6']

    model = initialize_model(args.weights, netDefinition, gpu_id=args.gpu_id, blobs_to_keep=blobs_to_keep)

    with NamedCudaScope(args.gpu_id):
        im_infer_keypoints(model, args.imgFolderDir, blob_input, blob_output_limb, blob_output_joint, args.imgSavePath, \
                           target_height=args.target_height, gpu_id=args.gpu_id)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    main(args)