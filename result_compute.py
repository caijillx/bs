# 0 person
# 1 bus
# 2 bicycle
# 3 car
# 4 motorbike

import json

file_name = "ten_fold_validation_result_faster_rcnn.json"
print(file_name)
result = json.load(open(file_name, 'r'))
for index, data in result.items():
    print("now validata %s" % index)
    cat_map = data["categlory_mAP"]
    voc_map = data["voc_mAP"]
    coco_map = data["coco_mAP"]
    if abs(sum(cat_map) / 5 - voc_map) < 1e-6:
        print("validation succeed")
    else:
        print(sum(cat_map) / 5)
        print(voc_map)
        print("validation failed")

all_voc_map = sum(data["voc_mAP"] for data in result.values()) / 10
all_coco_map = sum(data["coco_mAP"] for data in result.values()) / 10
person_map = sum(data["categlory_mAP"][0] for data in result.values()) / 10
bus_map = sum(data["categlory_mAP"][1] for data in result.values()) / 10
bicycle_map = sum(data["categlory_mAP"][2] for data in result.values()) / 10
car_map = sum(data["categlory_mAP"][3] for data in result.values()) / 10
motorbike_map = sum(data["categlory_mAP"][4] for data in result.values()) / 10
print("all_voc_map:", all_voc_map)
print("all_coco_map:", all_coco_map)
print("person_map:", person_map)
print("bus_map:", bus_map)
print("bicycle_map:", bicycle_map)
print("car_map:", car_map)
print("motorbike_map:", motorbike_map)
