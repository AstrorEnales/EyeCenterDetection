#!/usr/bin/env python

from PIL import Image, ImageDraw, ImageFont
import os

font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 16)

def draw_label(draw, position, text):
    draw.text((position[0] + 1, position[1] + 1), text, (255, 255, 255), font=font)
    draw.text(position, text, (0, 0, 0), font=font)

# BioID_0000_ascend_result
for i in range(0, 1521):
    im_ascend = Image.open('bioid/BioID_%04i_ascend_result.png' % i)
    im_ascendfit = Image.open('bioid/BioID_%04i_ascendfit_result.png' % i)
    im_paul = Image.open('bioid/BioID_%04i_paul_result.png' % i)
    im_naive = Image.open('bioid/BioID_%04i_naive_result.png' % i)
    im_evol = Image.open('bioid/BioID_%04i_evol_result.png' % i)
    
    im = Image.new('RGB', (im_naive.size[0] * 2, im_naive.size[1] * 3))
    draw = ImageDraw.Draw(im)
    
    im.paste(im_ascend, (0, 0))
    draw_label(draw, (5, 5), 'Ascend')
    
    im.paste(im_paul, (im_naive.size[0], 0))
    draw_label(draw, (im_naive.size[0] + 5, 5), 'Paul')
    
    im.paste(im_ascendfit, (0, im_naive.size[1]))
    draw_label(draw, (5, im_naive.size[1] + 5), 'Ascendfit')
    
    im.paste(im_naive, (im_naive.size[0], im_naive.size[1]))
    draw_label(draw, (im_naive.size[0] + 5, im_naive.size[1] + 5), 'Naive')
    
    im.paste(im_evol, (0, im_naive.size[1] * 2))
    draw_label(draw, (5, im_naive.size[1] * 2 + 5), 'Evol')
    
    im.save('movie/%04i.png' % i, 'PNG')
