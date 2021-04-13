# -*- coding:utf-8 -*-

# 本文件用于统计数据集各类别数目信息
##############################
# bus:2590个
# car:25317个
# person:11366个
# bicycle:698个
# motorbike:1232个
##############################
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
from PIL import Image


def parse_obj(xml_path, filename):
    tree = ET.parse(xml_path + filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        if obj.find('difficult').text == "0":
            obj_struct['name'] = obj.find('name').text
            objects.append(obj_struct)

    print(filename, objects)
    return objects


def read_image(image_path, filename):
    im = Image.open(image_path + filename)
    W = im.size[0]
    H = im.size[1]
    area = W * H
    im_info = [W, H, area]
    return im_info


if __name__ == '__main__':
    xml_path = r'/Users/llx/Desktop/mmdetection-master/data/VOCdevkit/VOC2007/Annotations/'
    filenamess = os.listdir(xml_path)
    filenames = []
    with open("/Users/llx/Desktop/mmdetection-master/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt", 'r') as f:
        xml_names = [i.strip() for i in f.readlines()]
    print("一共有{}个xml", len(xml_names))

    print("总共有{}个文件", len(filenamess))
    for name in filenamess:
        name = name.replace('.xml', '')
        if name in xml_names:
            filenames.append(name)
    print("一共有{}个name", len(filenames))
    recs = {}
    obs_shape = {}
    classnames = []
    num_objs = {}
    obj_avg = {}
    for i, name in enumerate(filenames):
        recs[name] = parse_obj(xml_path, name + '.xml')
    for name in filenames:
        for object in recs[name]:
            if object['name'] not in num_objs.keys():
                num_objs[object['name']] = 1

            else:
                num_objs[object['name']] += 1
            if object['name'] not in classnames:
                classnames.append(object['name'])
    for name in classnames:
        print('{}:{}个'.format(name, num_objs[name]))
    print('信息统计算完毕。')

# train
# bus:2348个
# car:22738个
# person:10253个
# bicycle:630个
# motorbike:1082个


# val
# car:2579个
# bus:242个
# person:1113个
# motorbike:150个
# bicycle:68个
