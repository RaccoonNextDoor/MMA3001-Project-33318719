"""Audit COCO archives, remove split leakage and export detection labels.

No source images or annotations are modified. Test data takes precedence over
validation, which takes precedence over training when duplicate groups occur.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import io
import json
import re
import zipfile
import numpy as np
from PIL import Image

CLASSES = ['loose-meat', 'packaging-error', 'twisted-meat', 'unsealed', 'wrinkle']


def source_key(name):
    """Return the pre-Roboflow filename used to group exposure variants."""
    return Path(name).name.split('.rf.')[0]


def clean_box(box, width, height):
    """Clip COCO xywh to the image; reject nonfinite or degenerate boxes.

    Args:
        box: Four pixel coordinates [x, y, width, height].
        width: Image width in pixels.
        height: Image height in pixels.
    Returns:
        Clipped xywh list.
    Raises:
        ValueError: Invalid image dimensions or unusable bounding box.
    """
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError('Invalid dimensions')
    x, y, w, h = map(float, box)
    if not np.isfinite([x,y,w,h]).all() or w <= 0 or h <= 0:
        raise ValueError('Nonfinite or nonpositive box')
    x1, y1 = max(0., x), max(0., y)
    x2, y2 = min(float(width), x+w), min(float(height), y+h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Box outside image')
    return [x1, y1, x2-x1, y2-y1]


def yolo_box(box, width, height):
    """Convert a valid COCO box into normalised centre-x, centre-y, w, h."""
    x,y,w,h = clean_box(box, width, height)
    return [(x+w/2)/width, (y+h/2)/height, w/width, h/height]


def assign_splits(rows):
    """Quarantine cross-split source or byte-identical duplicates transitively.

    Returns one split or 'excluded' per input record. Does not redistribute
    held-out observations into training. Same-split augmentations are retained.
    """
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i,r in enumerate(rows):
        for key in [('source',r['source']), ('hash',r['sha256'])]:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    rank = {'train':0, 'valid':1, 'test':2}
    highest = {}
    for i,r in enumerate(rows):
        k=root(i)
        highest[k] = max(highest.get(k,0),rank[r['original_split']])
    return [r['original_split'] if rank[r['original_split']]==highest[root(i)]
            else 'excluded' for i,r in enumerate(rows)]


def prepare(archive, out):
    """Validate/decode all images and export an audited, source-disjoint dataset.

    Args:
        archive: Path to original Roboflow COCO ZIP.
        out: New output directory. Existing manifests are rejected.
    Returns:
        Audit report dictionary; also writes manifest.json, audit.json and data.yaml.
    Raises:
        ValueError: Corrupt images, unsupported classes or invalid annotations.
        FileExistsError: Output already prepared.
    """
    archive,out = Path(archive),Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; clipped=0; original={}
    with zipfile.ZipFile(archive) as z:
        for split in ['train','valid','test']:
            data=json.loads(z.read(f'{split}/_annotations.coco.json'))
            categories={c['id']:c['name'] for c in data['categories']}
            ids=[i['id'] for i in data['images']]
            if len(set(ids))!=len(ids): raise ValueError('Duplicate image IDs')
            anns=defaultdict(list)
            for a in data['annotations']:
                if a['image_id'] not in set(ids): raise ValueError('Orphan annotation')
                if a['category_id'] not in categories: raise ValueError('Unknown category')
                anns[a['image_id']].append(a)
            original[split]={'images':len(ids),'annotations':len(data['annotations']),
                             'categories':dict(Counter(categories[a['category_id']] for a in data['annotations']))}
            for i in data['images']:
                name=i['file_name']
                if Path(name).name != name or '\\' in name: raise ValueError('Unsafe image path')
                payload=z.read(f'{split}/{name}')
                with Image.open(io.BytesIO(payload)) as im:
                    im.load()
                    if im.size != (i['width'],i['height']): raise ValueError('Image size mismatch')
                boxes=[]
                for a in anns[i['id']]:
                    category=categories[a['category_id']]
                    if category not in CLASSES: raise ValueError('Unsupported class')
                    box=clean_box(a['bbox'],i['width'],i['height'])
                    clipped += not np.allclose(box,a['bbox'])
                    boxes.append({'class_id':CLASSES.index(category),'bbox':box})
                rows.append({'file_name':name,'source':source_key(name),'original_split':split,
                             'sha256':hashlib.sha256(payload).hexdigest(),'width':i['width'],
                             'height':i['height'],'boxes':boxes,'count':len(boxes)})
        assigned=assign_splits(rows)
        for r,s in zip(rows,assigned):
            r['split']=s
            if s=='excluded': continue
            image_dir=out/'images'/s; label_dir=out/'labels'/s
            image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
            (image_dir/r['file_name']).write_bytes(z.read(f"{r['original_split']}/{r['file_name']}"))
            lines=[str(b['class_id'])+' '+' '.join(f'{v:.8f}' for v in yolo_box(b['bbox'],r['width'],r['height'])) for b in r['boxes']]
            (label_dir/(Path(r['file_name']).stem+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '')
    audit={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'original':original,
           'retained':{},'excluded_images':sum(s=='excluded' for s in assigned),'clipped_boxes':int(clipped),
           'classes':CLASSES,'policy':'Test > validation > training; connected components of source filename and byte hash',
           'limitations':['Source filenames are inferred groups, not verified tray IDs.',
                          'Adjacent frames may show the same tray; source-disjoint is not tray-disjoint.',
                          'No annotation means no labelled defect, not certified product safety.',
                          'Wrinkle is absent from original validation and test annotations.']}
    for s in ['train','valid','test']:
        subset=[r for r in rows if r['split']==s]
        audit['retained'][s]={'images':len(subset),'sources':len(set(r['source'] for r in subset)),
                             'annotations':sum(r['count'] for r in subset),
                             'empty':sum(r['count']==0 for r in subset)}
    (out/'manifest.json').write_text(json.dumps(rows,indent=2))
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    # JSON quoted strings are valid YAML scalars; forward slashes support Windows.
    yaml=f'path: {json.dumps(out.resolve().as_posix())}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n'
    yaml+=''.join(f'  {i}: {json.dumps(c)}\n' for i,c in enumerate(CLASSES))
    (out/'data.yaml').write_text(yaml)
    return audit
