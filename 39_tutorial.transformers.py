# # Tutorial - Transformers on DDP (Distributed Data Parralel)
# > An example of how to incorporate the transfomers library from HuggingFace with fastai

# code from
# https://github.com/fastai/fastai2/blob/master/nbs/39_tutorial.transformers.ipynb
# https://github.com/fastai/fastai2/blob/master/nbs/examples/train_imagenette.py

# From Pierre Guillou (28/06/2020): I changed the smallest number of codes from the file train_imagenette.py 
# in order to replicate the good running on a machine with 2 GPUs NVIDIA V100 32 RAM
# of the command: python -m fastai2.launch train_imagenette.py

# However, the same command with this current file (python -m fastai2.launch 39_tutorial.transformers.py) 
# waits for ever after the following dislay in a terminal:

# Run: 0
# Run: 0
# Training in distrib_ctx context on GPU 1
# Training in distrib_ctx context on GPU 0
# epoch     train_loss  valid_loss  perplexity  time
# Epoch 1/1 : |--------------------| 0.00% [0/38 00:00<00:00]

# When I do CTRL+C in the terminal, the last lines are 

# Traceback (most recent call last):
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/runpy.py", line 193, in _run_module_as_main
#     "__main__", mod_spec)
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/runpy.py", line 85, in _run_code
#     exec(code, run_globals)
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/site-packages/fastai2/launch.py", line 9, in <module>
#     args:Param("Args to pass to script", nargs='...', opt=False)=''
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/site-packages/fastscript/core.py", line 76, in call_parse
#     return _f()
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/site-packages/fastscript/core.py", line 73, in _f
#     func(**args.__dict__)
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/site-packages/fastai2/launch.py", line 26, in main
#     for process in processes: process.wait()
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/subprocess.py", line 1032, in wait
#     self._wait(timeout=sigint_timeout)
#   File "/mnt/home/pierre/.conda/envs/fastai2/lib/python3.7/subprocess.py", line 1647, in _wait
#     time.sleep(delay)
# KeyboardInterrupt

from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai2.vision.models.xresnet import *
from fastai2.callback.mixup import *
from fastscript import *

from fastai2.text.all import * #new

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

# GPT2
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
pretrained_weights = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
model = GPT2LMHeadModel.from_pretrained(pretrained_weights)

# datasets
path = untar_data(URLs.WIKITEXT_TINY)
df_train = pd.read_csv(path/'train.csv', header=None)
df_valid = pd.read_csv(path/'test.csv', header=None)
all_texts = np.concatenate([df_train[0].values, df_valid[0].values])

# tokenbizer fastai v2
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

splits = [list(range_of(df_train)), list(range(len(df_train), len(all_texts)))]
tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)

# get dataloaders
def get_dls(bs, sl, workers=None):
    if workers is None: workers = min(8, num_cpus())
    return tls.dataloaders(bs=bs, seq_len=sl, num_workers=workers)

class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    woof:  Param("Use imagewoof (otherwise imagenette)", int)=0,
    lr:    Param("Learning rate", float)=1e-2,
    size:  Param("Size (px: 128,192,256)", int)=128,
    sqrmom:Param("sqr_mom", float)=0.99,
    mom:   Param("Momentum", float)=0.9,
    eps:   Param("epsilon", float)=1e-6,
    epochs:Param("Number of epochs", int)=5,
    bs:    Param("Batch size", int)=8,
    sl:    Param("Sequence length", int)=1024,
    mixup: Param("Mixup", float)=0.,
    opt:   Param("Optimizer (adam,rms,sgd,ranger)", str)='ranger',
    arch:  Param("Architecture", str)=model,
    sh:    Param("Random erase max proportion", float)=0.,
    sa:    Param("Self-attention", int)=0,
    sym:   Param("Symmetry for self-attention", int)=0,
    beta:  Param("SAdam softplus beta", float)=0.,
    act_fn:Param("Activation function", str)='Mish',
    fp16:  Param("Use mixed precision training", int)=0,
    pool:  Param("Pooling method", str)='AvgPool',
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    meta:  Param("Metadata (ignored)", str)=''
):
    "Training of GPT2."

    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)
    if   opt=='adam'  : opt_func = partial(Adam, mom=mom, sqr_mom=sqrmom, eps=eps)
    elif opt=='rms'   : opt_func = partial(RMSprop, sqr_mom=sqrmom)
    elif opt=='sgd'   : opt_func = partial(SGD, mom=mom)
    elif opt=='ranger': opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    # dataloaders
    dls = get_dls(bs, sl)
    # if not gpu: print(f'epochs: {epochs}; lr: {lr}; size: {size}; sqrmom: {sqrmom}; mom: {mom}; eps: {eps}')

    # m,act_fn,pool = [globals()[o] for o in (arch,act_fn,pool)]

    for run in range(runs):
        print(f'Run: {run}')
        learn = Learner(dls, model, 
                        loss_func=CrossEntropyLossFlat(), 
                        cbs=[DropOutput], metrics=Perplexity())
        
        # if dump: print(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        # cbs = MixUp(mixup) if mixup else []

        n_gpu = torch.cuda.device_count()

        # The old way to use DataParallel, or DistributedDataParallel training:
        # if gpu is None and n_gpu: learn.to_parallel()
        # if num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai2.launch`

        # the context manager way of dp/ddp, both can handle single GPU base case.
        ctx = learn.parallel_ctx if gpu is None and n_gpu else learn.distrib_ctx

        with partial(ctx, gpu)(): # distributed traing requires "-m fastai2.launch"
            print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
            # learn.fit_flat_cos(epochs, lr, wd=1e-2, cbs=cbs)
            learn.fit_one_cycle(1, 1e-4)