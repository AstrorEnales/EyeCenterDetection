#!/usr/bin/env python

import urllib.request
import os
import zipfile
from sys import argv
import subprocess
import operator
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

bioidfile = 'BioID-FaceDatabase-V1.2.zip'
bioiddir = 'bioid'
bioidptsfile = 'bioid_pts.zip'
bioidptsdir = 'bioidpts'
eyeCenterApp = '..\\src\\build\\RelWithDebInfo\\EyeCenter.exe' if len(argv) <= 1 else argv[2]

# Load the bioid database
if not os.path.isdir(bioiddir):
    if not os.path.isfile(bioidfile):
        urllib.request.urlretrieve('https://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip', bioidfile)
  
    with zipfile.ZipFile(bioidfile, 'r') as z:
        z.extractall(bioiddir)
    
if not os.path.isdir(bioidptsdir):
    if not os.path.isfile(bioidptsfile):
        urllib.request.urlretrieve('https://ftp.uni-erlangen.de/pub/facedb/bioid_pts.zip', bioidptsfile)
    with zipfile.ZipFile(bioidptsfile, 'r') as z:
        z.extractall(bioidptsdir)

testCases = {}

for file in os.listdir(bioiddir):
    if file.endswith('.eye'):
        with open(os.path.join(bioiddir, file)) as f:
            positions = [int(x.strip()) for x in f.readlines()[1].split('\t') if len(x.strip()) > 0]
            positions = [(positions[0], positions[1]), (positions[2], positions[3])]
            testCases[file.replace('.eye', '')] = positions


def preprocess_image(id):
    '''
    0 = right eye pupil
    1 = left eye pupil
    2 = right mouth corner
    3 = left mouth corner
    4 = outer end of right eye brow
    5 = inner end of right eye brow
    6 = inner end of left eye brow
    7 = outer end of left eye brow
    8 = right temple
    9 = outer corner of right eye
    10 = inner corner of right eye
    11 = inner corner of left eye
    12 = outer corner of left eye
    13 = left temple
    14 = tip of nose
    15 = right nostril
    16 = left nostril
    17 = centre point on outer edge of upper lip
    18 = centre point on outer edge of lower lip
    19 = tip of chin
    '''
    with open(os.path.join(bioidptsdir, 'points_20\\%s.pts' % id)) as f:
        positions = [[int(float(y)) for y in x.strip().split(' ')] for x in f.readlines()[3:-1]]
        rect = (positions[8][0], positions[4][1], positions[13][0], positions[14][1])
        im = Image.open(os.path.join(bioiddir, '%s.pgm' % id))
        outfile = os.path.join(bioiddir, '%s.png' % id)
        im.crop(rect).save(outfile, 'PNG')
        return [outfile, rect]


def draw_cross(draw, x, y, color):
    draw.line((x - 5, y, x + 5, y), fill=color)
    draw.line((x, y - 5, x, y + 5), fill=color)


def draw_result_image(id, predictions):
    im = Image.open(os.path.join(bioiddir, '%s.pgm' % id)).convert('RGB')
    draw = ImageDraw.Draw(im)
    left = testCases[id][0]
    right = testCases[id][1]
    draw_cross(draw, left[0], left[1], 'green')
    draw_cross(draw, right[0], right[1], 'green')
    for pred in predictions:
        draw_cross(draw, pred[0], pred[1], 'red')
    outfile = os.path.join(bioiddir, '%s_result.png' % id)
    im.save(outfile, 'PNG')


def run_eye_center(id, mode):
    result = subprocess.check_output([eyeCenterApp, '-m', mode, '-s', '-i', os.path.join(bioiddir, '%s.png' % id)]).decode('utf-8')
    result = [x.strip() for x in result.replace('\r', '').split('\n') if len(x.strip()) > 0]
    # result[0] is the time used for the detection mode. The following lines are the found targets
    return float(result[0]), [[int(y) for y in x.split('\t')] for x in result[1:]]


def calc_error(id, predictions):
    # Divide the error by the distance between the
    # two eye centers to normalize different face sizes
    norm_distance = math.sqrt((testCases[id][0][0] - testCases[id][1][0])**2 + (testCases[id][0][1] - testCases[id][1][1])**2)
    result = []
    for pred in predictions:
        distances = [math.sqrt((pred[0] - real[0])**2 + (pred[1] - real[1])**2) for real in testCases[id]]
        min_index, min_value = min(enumerate(distances), key=operator.itemgetter(1))
        result.append(min_value / norm_distance)
    return result

modes = ['naive', 'ascend']
results = {x: [] for x in modes}
for id in list(testCases.keys())[0:10]:
    cutout = preprocess_image(id)
    try:
        for mode in modes:
            predictions = run_eye_center(id, mode)
            time_used = predictions[0]
            predictions = predictions[1]
            predictions = [[cutout[1][0] + x[0], cutout[1][1] + x[1]] for x in predictions]
            error = calc_error(id, predictions)
            results[mode].append([time_used, error])
            draw_result_image(id, predictions)
    finally:
        os.remove(cutout[0])

fig, axes = plt.subplots(nrows=len(modes), ncols=2)
for i in range(len(modes)):
    axes[i, 0].set_title('Mode: %s' % modes[i])

    axes[i, 0].set_xlabel('Execution time [s]')
    axes[i, 0].hist([x[0] for x in results[modes[i]]])

    axes[i, 1].set_xlabel('Normalized error')
    error_data = [item for sublist in results[modes[i]] for item in sublist[1]]
    axes[i, 1].hist(error_data)

fig.tight_layout()
fig.savefig('result_hist.png')
plt.show()
