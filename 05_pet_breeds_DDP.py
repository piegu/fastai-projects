# source: https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb
# source: https://github.com/fastai/fastai2/blob/master/nbs/examples/train_imagenette.py

from utils import *    
from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastscript import *

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_dls(bs, workers=None):
    path = untar_data(URLs.PETS)
    #source = untar_data(path)
    if workers is None: workers = min(8, num_cpus())
    batch_tfms = [aug_transforms(size=224, min_scale=0.75)]
    dblock = DataBlock(blocks = (ImageBlock, CategoryBlock),
                       get_items=get_image_files, 
                       splitter=RandomSplitter(seed=42),
                       get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                       item_tfms=Resize(460),
                       batch_tfms=aug_transforms(size=224, min_scale=0.75))
    return dblock.dataloaders(path/"images", bs=bs, num_workers=workers)

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    bs:    Param("Batch size", int)=64,
    arch:  Param("Architecture", str)=resnet34,
    runs:  Param("Number of times to repeat training", int)=1
):
    "Training of pets."
    
    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)
        
    dls = get_dls(bs)

    for run in range(runs):
        print(f'Run: {run}')
        
        learn = cnn_learner(dls, arch, metrics=error_rate).to_fp16()

        n_gpu = torch.cuda.device_count()
        
        # The old way to use DataParallel, or DistributedDataParallel training:
        # if gpu is None and n_gpu: learn.to_parallel()
        # if num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai2.launch`

        # the context manager way of dp/ddp, both can handle single GPU base case.
        ctx = learn.parallel_ctx if gpu is None and n_gpu else learn.distrib_ctx

        with partial(ctx, gpu)(): # distributed traing requires "-m fastai2.launch"
            print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
            learn.fine_tune(1)