from PIL import Image, ImageDraw
import math
import numpy as np
from sys import argv

img = Image.open(argv[1]).convert('L')

#img.thumbnail((512, 512), Image.ANTIALIAS)

pixels = list(img.getdata())

N = img.width * img.height

print('Image: %sx%s' % (img.width, img.height))

print('Build difference map')
# ABAB|A
# BABA|A
# ----+-
# BBBB|x
diff = np.zeros(N)
for i in range(0, N):
  pixel_diff = 0
  p = pixels[i]
  
  i_prev = i - img.width
  if i_prev >= 0:
    # Add quadratic difference of pixel and previous pixel in y
    d = p - pixels[i_prev]
    # Add A
    pixel_diff += diff[i_prev] + d * d
  
  if i % img.width > 0:
    # Add B and Subtract overlap of A and B
    pixel_diff += diff[i - 1] - (diff[i_prev - 1] if i >= img.width else 0)
    # Add quadratic difference of pixel and previous pixel in x
    d = p - pixels[i - 1]
    pixel_diff += d * d
  
  diff[i] = pixel_diff

max_diff = max(diff)
min_diff = min(diff)

print('Process image regions')
result = img.convert('RGBA')
layer = Image.new('RGBA', result.size, (255, 255, 255, 0))

draw = ImageDraw.Draw(layer, 'RGBA')

def getAreaRoughness(x1, y1, x2, y2):
  x2 = min(img.width - 1, x2)
  y2 = min(img.height - 1, y2)
  r = diff[y2 * img.width + x2]
  if y1 > 0:
    r -= diff[(y1 - 1) * img.width + x2]
  if x1 > 0:
    r -= diff[(y2 - 1) * img.width + x1 - 1]
  if x1 > 0 and y1 > 0:
    r += diff[(y1 - 1) * img.width + x1 - 1]
  return r#0 if r == 0 else math.log(r)

def quadTree(depth, x1, y1, x2, y2, result_areas):
  wh = int((x2 + x1) / 2)
  hh = int((y2 + y1) / 2)
  
  if wh - x1 == 0 or hh - y1 == 0:
    result_areas.append((x1, y1, x2, y2, getAreaRoughness(x1, y1, x2, y2)))
    return
  
  value_tl = getAreaRoughness(x1, y1, wh, hh)
  value_tr = getAreaRoughness(wh + 1, y1, x2, hh)
  value_bl = getAreaRoughness(x1, hh + 1, wh, y2)
  value_br = getAreaRoughness(wh + 1, hh + 1, x2, y2)
  
  # TODO: improve decision
  thresh = (max_diff - min_diff) * 0.01
  if value_tl > thresh or value_tr > thresh or value_bl > thresh or value_br > thresh:
    quadTree(depth + 1, x1, y1, wh, hh, result_areas)
    quadTree(depth + 1, x1, hh + 1, wh, y2, result_areas)
    quadTree(depth + 1, wh + 1, y1, x2, hh, result_areas)
    quadTree(depth + 1, wh + 1, hh + 1, x2, y2, result_areas)
  else:
    result_areas.append((x1, y1, x2, y2, getAreaRoughness(x1, y1, x2, y2)))


tree_areas = []
quadTree(1, 0, 0, img.width - 1, img.height - 1, tree_areas)

#print(tree_areas)

print('Draw output')
min_r = min([x[4] for x in tree_areas])
max_r = max([x[4] for x in tree_areas])
factor = 255.0 / max(1, max_r - min_r)

for area in tree_areas:
  r = int((area[4] - min_r) * factor)
  draw.rectangle(area[0:4], (r, 0, 0, 200))
  #draw.rectangle(area[0:4], outline='green')

out = Image.alpha_composite(result, layer)
out.show()
