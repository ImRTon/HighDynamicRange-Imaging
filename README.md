# VFX Project1 HDR
Calculate HDR image using multi-exposure images.

We implemented `Image Alignment using MTB`, `Solve Response Curve`, `Radiance Map` and `Tone Mapping`.

## Environment
```
Python 3.8
    opencv-python
    Pillow
    numpy
    tqdm
    exifread
    matplotlib
    ...
```

## Install
```
conda create --name VFX python=3.8
conda activate VFX
pip install -r requirements.txt
```

## Run
-i --input_dir INPUT_DIR
-a --align_img `True` / `False`
-s --sample_method `uniform` / `random`
```
python main.py -i INPUT_DIR -a ALIGN_IMG_OR_NOT -s SAMPLE_METHOD
```
