import os
import json
import imagesize
from dassl.data.datasets import DATASET_REGISTRY, DatumWithBbox, DatasetBase

from .hico_text_label import hico_hoi_only_text_label


@DATASET_REGISTRY.register()
class HICO_DET(DatasetBase):

    dataset_dir = "hico_20160224_det"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_image_dir = os.path.join(self.dataset_dir, "images", "train2015")
        self.test_image_dir = os.path.join(self.dataset_dir, "images", "test2015")
        with open(os.path.join(self.dataset_dir, "annotations", "trainval_hico.json"), "r") as f:
            self.train_anno_file = json.load(f)
        with open(os.path.join(self.dataset_dir, "annotations", "test_hico.json"), "r") as f:
            self.test_anno_file = json.load(f)

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))
        
        self.crop_scale = cfg.crop_scale

        train = self.read_traindata(self.train_image_dir, self.train_anno_file)
        test = self.read_testdata(self.test_image_dir, self.test_anno_file)

        super().__init__(train_x=train, val=None, test=test)

    def read_traindata(self, image_dir, annotations):
        items = []
        for img_anno in annotations:
            bboxes = img_anno['annotations']
            imname = img_anno["file_name"]
            impath = os.path.join(image_dir, imname)
            imsize = imagesize.get(impath) # (width, height)
            for hoi in img_anno['hoi_annotation']:
                label = hoi['hoi_category_id'] - 1
                classname = list(hico_hoi_only_text_label.values())[label]
                human_bbox = bboxes[hoi['subject_id']]['bbox']
                object_bbox = bboxes[hoi['object_id']]['bbox']
                union_bbox = [
                    min(human_bbox[0], object_bbox[0]),
                    min(human_bbox[1], object_bbox[1]),
                    max(human_bbox[2], object_bbox[2]),
                    max(human_bbox[3], object_bbox[3])]
                union_bbox = self.scale_bbox(union_bbox, imsize)
                item = DatumWithBbox(impath=impath, label=label, classname=classname, bbox=union_bbox)
                items.append(item)

        return items

    def read_testdata(self, image_dir, annotations):
        items = []
        for img_anno in annotations:
            bboxes = img_anno['annotations']
            imname = img_anno["file_name"]
            impath = os.path.join(image_dir, imname)
            imsize = imagesize.get(impath) # (width, height)
            for hoi in img_anno['hoi_annotation']:
                label_tuple = (hoi['category_id'] - 1, self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']))
                label = list(hico_hoi_only_text_label.keys()).index(label_tuple)
                classname = list(hico_hoi_only_text_label.values())[label]
                human_bbox = bboxes[hoi['subject_id']]['bbox']
                object_bbox = bboxes[hoi['object_id']]['bbox']
                union_bbox = [
                    min(human_bbox[0], object_bbox[0]),
                    min(human_bbox[1], object_bbox[1]),
                    max(human_bbox[2], object_bbox[2]),
                    max(human_bbox[3], object_bbox[3])]
                union_bbox = self.scale_bbox(union_bbox, imsize)
                item = DatumWithBbox(impath=impath, label=label, classname=classname, bbox=union_bbox)
                items.append(item)

        return items

    def scale_bbox(self, box, imsize):
        x1, y1, x2, y2 = box

        w = x2 - x1
        h = y2 - y1

        area = w * h
        scaled_area = area * self.crop_scale
        scale_factor = (scaled_area / area) ** 0.5

        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        new_x1 = max(0, int(x1 - ((new_w - w) / 2)))
        new_y1 = max(0, int(y1 - ((new_h - h) / 2)))
        new_x2 = min(imsize[0], new_x1 + new_w)
        new_y2 = min(imsize[1], new_y1 + new_h)

        return [new_x1, new_y1, new_x2, new_y2]