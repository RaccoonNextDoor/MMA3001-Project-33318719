"""Optional GPU detector workflow using Ultralytics YOLOv8n.

Train/validation only by default. A separate explicit final-test command exists
so test performance cannot accidentally guide training or threshold selection.
"""
import argparse
from pathlib import Path
import json


def main():
    """Train, validate or predict with explicit dataset/weight arguments."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','validate','predict','final-test'])
    p.add_argument('--data',default='data/prepared/data.yaml')
    p.add_argument('--weights',default='yolov8n.pt')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='0')
    p.add_argument('--name',default='yolov8n_640')
    p.add_argument('--source')
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--confirm-frozen-model',action='store_true')
    args=p.parse_args()
    if args.mode=='final-test' and not args.confirm_frozen_model:
        p.error('Freeze model/settings first, then pass --confirm-frozen-model.')
    if args.mode!='train' and args.weights=='yolov8n.pt':
        p.error('Specify your trained --weights path, not generic COCO weights.')
    if args.mode=='predict' and not args.source: p.error('--source required')
    from ultralytics import YOLO
    model=YOLO(args.weights)
    if args.mode=='train':
        model.train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
                    device=args.device,workers=0,seed=42,deterministic=True,
                    project='runs/detect',name=args.name,exist_ok=False,patience=15,
                    mosaic=0.,mixup=0.,fliplr=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
                    translate=0.,scale=0.,degrees=0.,shear=0.,perspective=0.)
    elif args.mode in ('validate','final-test'):
        results=model.val(data=args.data,split='test' if args.mode=='final-test' else 'val',
                          imgsz=args.imgsz,batch=args.batch,device=args.device,workers=0,
                          project='runs/evaluate',name=args.name,exist_ok=False,
                          plots=True,save_json=True)
        metrics={k:float(v) for k,v in results.results_dict.items()}
        print(json.dumps(metrics,indent=2))
        (Path(results.save_dir)/'metrics.json').write_text(json.dumps(metrics,indent=2))
    else:
        model.predict(source=args.source,imgsz=args.imgsz,device=args.device,
                      conf=args.conf,save=True,save_txt=True,save_conf=True,
                      project='runs/predict',name=args.name,exist_ok=False)

if __name__=='__main__': main()
