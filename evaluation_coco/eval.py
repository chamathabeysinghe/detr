from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

annType = 'bbox'
cocoGt = COCO('./ground-truth-test.json')
cocoDt = cocoGt.loadRes('./eval-results.json')
cocoEval = COCOeval(cocoGt, cocoDt, annType)
imgIds = sorted(cocoGt.getImgIds())
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

