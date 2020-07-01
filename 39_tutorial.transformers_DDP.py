# # Tutorial - Transformers on DDP (Distributed Data Parralel)
# > An example of how to incorporate the transfomers library from HuggingFace with fastai

# code from
# https://github.com/fastai/fastai2/blob/master/nbs/39_tutorial.transformers.ipynb
# source: https://github.com/fastai/fastai2/blob/master/nbs/examples/train_imdbclassifier.py

# From Pierre Guillou (01/07/2020): I changed the smallest number of codes from the file train_imdbclassifier.py
# in order to replicate the good running on a machine with 2 GPUs NVIDIA V100 32 RAM
# of the command: python -m fastai2.launch train_imdbclassifier.py

# However, the same command with this current file (python -m fastai2.launch 39_tutorial.transformers_DDP.py) 
# did not work.

# transformers==3.0.0
# fastai2==0.0.17

from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.distributed import *
from fastprogress import fastprogress
from fastai2.callback.mixup import *
from fastscript import *
from fastai2.text.all import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    lr:    Param("base Learning rate", float)=1e-4,
    bs:    Param("Batch size", int)=8,
    sl:    Param("Sequence length", int)=1024,
    epochs:Param("Number of epochs", int)=1,
    fp16:  Param("Use mixed precision training", int)=0,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    "Training of IMDB classifier."

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if gpu is None: gpu = list(range(n_gpu))[0] 
        torch.cuda.set_device(gpu)
    else:
        n_gpu = None

    # GPT2
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    pretrained_weights = 'gpt2'
    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
    model = GPT2LMHeadModel.from_pretrained(pretrained_weights)

    # datasets
    path = rank0_first(lambda: untar_data(URLs.WIKITEXT_TINY))
    df_train = pd.read_csv(path/'train.csv', header=None)
    df_valid = pd.read_csv(path/'test.csv', header=None)
    all_texts = np.concatenate([df_train[0].values, df_valid[0].values])

    # fastai v2 tokenizer
    class TransformersTokenizer(Transform):
        def __init__(self, tokenizer): self.tokenizer = tokenizer
        def encodes(self, x): 
            toks = self.tokenizer.tokenize(x)
            return tensor(self.tokenizer.convert_tokens_to_ids(toks))
        def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))
    splits = [list(range_of(df_train)), list(range(len(df_train), len(all_texts)))]
    tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
    
    # get dataloaders
    workers = min(8, num_cpus())
    dls = tls.dataloaders(bs=bs, seq_len=sl, num_workers=workers)
    
    class DropOutput(Callback):
        def after_pred(self): self.learn.pred = self.pred[0]

    for run in range(runs):
        print(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}; sl: {sl}')

        learn = rank0_first(lambda: Learner(dls, model, loss_func=CrossEntropyLossFlat(), 
                                            cbs=[DropOutput], metrics=Perplexity()))

        if dump: print(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        # TODO: DataParallel would hit floating point error, disabled for now.
        # if gpu is None and n_gpu: ctx = partial(learn.parallel_ctx, device_ids=list(range(n_gpu)))

        # Workaround: In PyTorch 1.4, need to set DistributedDataParallel() with find_unused_parameters=True,
        # to avoid a crash that only happens in distributed mode of text_classifier_learner.fine_tune()

        # if num_distrib() > 1 and torch.__version__.startswith("1.4"): DistributedTrainer.fup = True
        DistributedTrainer.fup = True

        with learn.distrib_ctx(cuda_id=gpu): # distributed traing requires "-m fastai2.launch"
            print(f"Training in distributed data parallel context on GPU {gpu}", flush=True)
            learn.fit_one_cycle(epochs, lr)
           