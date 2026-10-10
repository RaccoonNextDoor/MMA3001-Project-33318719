"""Analytical checks for label conversion and cross-split leakage handling."""
import pytest
from rasher.data import clean_box,yolo_box,assign_splits,source_key


def test_box_conversion_known_geometry():
    assert yolo_box([10,20,40,20],100,100)==pytest.approx([.3,.3,.4,.2])


def test_clipping_preserves_visible_area():
    assert clean_box([-5,90,20,20],100,100)==[0,90,15,10]


@pytest.mark.parametrize('box',[[0,0,0,10],[0,0,-1,1],[101,0,2,2],[0,0,float('nan'),1]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):clean_box(box,100,100)


def test_variant_group():
    assert source_key('frame_12_jpg.rf.abcdef.jpg')=='frame_12_jpg'


def test_transitive_leakage():
    rows=[{'source':'a','sha256':'x','original_split':'train'},
          {'source':'a','sha256':'y','original_split':'valid'},
          {'source':'b','sha256':'y','original_split':'test'},
          {'source':'c','sha256':'z','original_split':'train'}]
    assert assign_splits(rows)==['excluded','excluded','test','train']


def test_train_augmentations_retained():
    rows=[{'source':'a','sha256':h,'original_split':'train'} for h in ['x','y']]
    assert assign_splits(rows)==['train','train']


def test_prepare_archive_end_to_end(tmp_path):
    import io,json,zipfile
    from PIL import Image
    from rasher.data import prepare
    zpath=tmp_path/'input.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        for index,split in enumerate(['train','valid','test']):
            name=f'frame{index}.rf.example.jpg'
            stream=io.BytesIO();Image.new('RGB',(100,100),(index*50,0,0)).save(stream,format='JPEG')
            z.writestr(f'{split}/{name}',stream.getvalue())
            d={'images':[{'id':0,'file_name':name,'width':100,'height':100}],
               'categories':[{'id':4,'name':'unsealed'}],
               'annotations':[{'id':0,'image_id':0,'category_id':4,'bbox':[10,20,40,20]}]}
            z.writestr(f'{split}/_annotations.coco.json',json.dumps(d))
    out=tmp_path/'prepared';report=prepare(zpath,out)
    assert report['excluded_images']==0
    assert report['retained']['test']['images']==1
    values=(out/'labels/train/frame0.rf.example.txt').read_text().split()
    assert values[0]=='3'
    assert list(map(float,values[1:]))==pytest.approx([.3,.3,.4,.2])
    with pytest.raises(FileExistsError):prepare(zpath,out)


def test_final_test_requires_explicit_freeze(monkeypatch):
    import sys
    from rasher.detector import main
    monkeypatch.setattr(sys,'argv',['detector','final-test','--weights','my-model.pt'])
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==2
