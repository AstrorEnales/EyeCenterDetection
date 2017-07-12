#!/usr/bin/env python

from sys import argv, version_info
python_version3 = version_info > (3, 0)
if python_version3: import urllib.request as urllibreq
else: import urllib as urllibreq
import os
import zipfile
import subprocess
import operator
import math
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

bioidfile = 'BioID-FaceDatabase-V1.2.zip'
bioiddir = 'bioid'
bioidptsfile = 'bioid_pts.zip'
bioidptsdir = 'bioidpts'
eyeCenterApp = '../src/build/RelWithDebInfo/EyeCenter.exe' if len(argv) <= 1 else argv[1]

# Load the bioid database
if not os.path.isdir(bioiddir):
    if not os.path.isfile(bioidfile):
        urllibreq.urlretrieve('https://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip', bioidfile)
  
    with zipfile.ZipFile(bioidfile, 'r') as z:
        z.extractall(bioiddir)
    
if not os.path.isdir(bioidptsdir):
    if not os.path.isfile(bioidptsfile):
        urllibreq.urlretrieve('https://ftp.uni-erlangen.de/pub/facedb/bioid_pts.zip', bioidptsfile)
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
    result = []
    with open(os.path.join(bioidptsdir, 'points_20/%s.pts' % id.lower())) as f:
        positions = [[int(float(y)) for y in x.strip().split(' ')] for x in f.readlines()[3:-1]]
        im = Image.open(os.path.join(bioiddir, '%s.pgm' % id))
        
        rect = (positions[8][0], positions[4][1], positions[14][0], positions[14][1])
        outfile = os.path.join(bioiddir, '%s_1.png' % id)
        im.crop(rect).save(outfile, 'PNG')
        result.append([outfile, rect])
        
        rect = (positions[14][0], positions[4][1], positions[13][0], positions[14][1])
        outfile = os.path.join(bioiddir, '%s_2.png' % id)
        im.crop(rect).save(outfile, 'PNG')
        result.append([outfile, rect])
        
    return result


def draw_cross(draw, x, y, color):
    draw.line((x - 5, y, x + 5, y), fill=color)
    draw.line((x, y - 5, x, y + 5), fill=color)


def draw_result_image(id, predictions, mode):
    im = Image.open(os.path.join(bioiddir, '%s.pgm' % id)).convert('RGB')
    draw = ImageDraw.Draw(im)
    left = testCases[id][0]
    right = testCases[id][1]
    draw_cross(draw, left[0], left[1], 'green')
    draw_cross(draw, right[0], right[1], 'green')
    for pred in predictions:
        draw_cross(draw, pred[0], pred[1], 'red')
    outfile = os.path.join(bioiddir, '%s_%s_result.png' % (id, mode))
    im.save(outfile, 'PNG')


def run_eye_center(id, num, mode):
    result = subprocess.check_output([eyeCenterApp, '-m', mode, '-s', '-i', os.path.join(bioiddir, '%s_%s.png' % (id, num))]).decode('utf-8')
    result = [x.strip() for x in result.replace('\r', '').split('\n') if len(x.strip()) > 0]
    # result[0] is the time used for the detection mode. The following line is the found target
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

modes = ['naive', 'ascend', 'ascendfit', 'paul', 'evol']
results = {x: [] for x in modes}
count = 0
keys = list(testCases.keys())
keys = keys[100:120] # For faster testing, limit the number of test images
for id in keys:
    count += 1
    print(count, '/', len(keys))
    cutouts = preprocess_image(id)
    try:
        for mode in modes:
            pred_positions = []
            for num in [1, 2]:
                predictions = run_eye_center(id, num, mode)
                cutout = cutouts[num - 1][1]
                pixel_count = (cutout[3] - cutout[1]) * (cutout[2] - cutout[0])
                time_used = predictions[0] * 1000.0 / pixel_count
                predictions = predictions[1]
                predictions = [[cutout[0] + x[0], cutout[1] + x[1]] for x in predictions]
                pred_positions.extend(predictions)
                error = calc_error(id, predictions)
                results[mode].append([time_used, error, predictions, id])
            draw_result_image(id, pred_positions, mode)
    finally:
        os.remove(cutouts[0][0])
        os.remove(cutouts[1][0])

# Save the results into a csv file for future analysis
for mode in modes:
  with open('results/evaluate_results_%s.csv' % mode, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['bioid', 'time', 'errors', 'predictions'])
      for result in results[mode]:
        errors_text = '|'.join([str(round(x, 6)) for x in result[1]])
        predictions_text = '|'.join(['%s;%s' % (x[0], x[1]) for x in result[2]])
        writer.writerow([result[3], round(result[0], 6), errors_text, predictions_text])

# Plot the results into histograms
fig, axes = plt.subplots(nrows=len(modes), ncols=2, figsize=(10, len(modes) * 2))
for i in range(len(modes)):
    axes[i, 0].set_title('Mode: %s' % modes[i])

    axes[i, 0].set_xlabel('Execution time [s / 1k pixel]')
    axes[i, 0].set_xlim(0, 0.5)
    axes[i, 0].hist([x[0] for x in results[modes[i]]])

    axes[i, 1].set_xlabel('Normalized error')
    axes[i, 1].set_xlim(0, 0.5)
    error_data = [item for sublist in results[modes[i]] for item in sublist[1]]
    axes[i, 1].hist(error_data)

fig.tight_layout()
fig.savefig('result_hist.png', dpi=300)
#plt.show()
