# fastai-projects
Jupyter notebooks that use the Fastai library

## fastai v2.0

### Faster than training from scratch — Fine-tuning the English GPT-2 in any language with Hugging Face and fastai v2 (practical case with Portuguese)

In this [notebook](https://github.com/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2.ipynb), instead of training from scratch, we will see how to fine-tune in just over a day, on one GPU and with a little more than 1GB of training data an English pre-trained transformer-based language model to any another language. As a practical case, we fine-tune to Portuguese the English pre-trained GPT-2 by wrapping the Transformers and Tokenizers libraries of Hugging Face into fastai v2. We thus create a new language model: [GPorTuguese-2](https://huggingface.co/pierreguillou/gpt2-small-portuguese), a language model for Portuguese text generation (and more NLP tasks...).

**Note**: as the full notebook is very detailed, use this [fast notebook](https://github.com/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2_FAST.ipynb) ([nbviewer version](https://nbviewer.jupyter.org/github/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2_FAST.ipynb)) if you just want to run the code without explanation.

![GPorTuguese-2 (Portuguese GPT-2 small) , a language model for Portuguese text generation (and more NLP tasks...)](https://github.com/piegu/fastai-projects/blob/master/images/hfmh.png "GPorTuguese-2 (Portuguese GPT-2 small) , a language model for Portuguese text generation (and more NLP tasks...)")

### Byte-level BPE, an universal tokenizer but...

In this study, we will see that, while it is true that a BBPE tokenizer (Byte-level Byte-Pair-Encoding) trained on a huge monolingual corpus can tokenize any word of any language (there is no unknown token), it requires on average almost 70% of additional tokens when it is applied to a text in a language different from that used for its training. This information is key when it comes to choosing a tokenizer to train a natural language model like a Transformer model.
- [post](https://medium.com/@pierre_guillou/byte-level-bpe-an-universal-tokenizer-but-aff932332ffe) in medium
- notebook [Byte-level-BPE_universal_tokenizer_but.ipynb](https://github.com/piegu/fastai-projects/blob/master/Byte-level-BPE_universal_tokenizer_but.ipynb) ([nbviewer version](https://nbviewer.jupyter.org/github/piegu/fastai-projects/blob/master/Byte-level-BPE_universal_tokenizer_but.ipynb))
- Wikipedia downloading functions in the file [nlputils_fastai2.py](https://github.com/piegu/fastai-projects/blob/master/nlputils_fastai2.py)

### Distributed Data Parallel (DDP)

The script [05_pet_breeds_DDP.py](https://github.com/piegu/fastai-projects/blob/master/05_pet_breeds_DDP.py) gives the code to run for training a Deep Learning model in Distributed Data Parallel (DDP) mode with fastai v2. It is inspired by the notebook [05_pet_breeds.ipynb](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb) from the [fastbook](https://github.com/fastai/fastbook/#the-fastai-book---draft) (fastai v2), the [Distributed and parallel training](https://dev.fast.ai/distributed#DistributedTrainer) fastai v2 documentation and the notebook [train_imagenette.py](https://github.com/fastai/fastai2/blob/master/nbs/examples/train_imagenette.py).

In order to get it run, you need to launch the following command within a fastai 2 virtual environment in a Terminal of a server with at least 2 GPUs:

`python -m fastai2.launch 05_pet_breeds_DDP.py`

### Data Parallel (DP)

The notebook [05_pet_breeds_DataParallel.ipynb](https://github.com/piegu/fastai-projects/blob/master/05_pet_breeds_DataParallel.ipynb) ([nbviewer version](https://nbviewer.jupyter.org/github/piegu/fastai-projects/blob/master/05_pet_breeds_DataParallel.ipynb)) gives the code to run for training a Deep Learning model in Data Parallel (DP) mode with PyTorch and fastai v2. It is inspired by the notebook [05_pet_breeds.ipynb](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb) from the [fastbook](https://github.com/fastai/fastbook/#the-fastai-book---draft) (fastai v2), the [Distributed and parallel training](https://dev.fast.ai/distributed#ParallelTrainer) fastai v2 documentation and the notebook [train_imagenette.py](https://github.com/fastai/fastai2/blob/master/nbs/examples/train_imagenette.py).

### How to create groups of layers and each one with a different Learning Rate?

The objective of this [notebook](https://github.com/piegu/fastai-projects/blob/master/groups_layers_different_learning_rates_fastaiv2.ipynb) ([nbviewer version](https://nbviewer.jupyter.org/github/piegu/fastai-projects/blob/master/groups_layers_different_learning_rates_fastaiv2.ipynb?flush_cache=true)) is to explain how to create parameters groups for a model with fastai v2 in order to train each one with a different learning rate, how to pass the list of Learning rates and how to check the Learning Rates effectively used by the Optimizer during the training.

### How fastai v2 deals with batch sizes for the training and validation datasets

The objective of this [notebook](https://github.com/piegu/fastai-projects/blob/master/fastaiv2_batchsize_training_validation_datasets.ipynb) is to explain how [fastai v2](https://github.com/fastai/fastai2) deals with batch sizes for the training and validation datasets.

### Comparison of sizes of learn.export() files by batch size
The objective of this [notebook](https://github.com/piegu/fastai-projects/blob/master/learn_export_files_by_batchsize_fastaiv2.ipynb) is to show that the sizes of pkl files created by learn.export() of fastai v2 are different depending on the batch size used. This is odd, no?

## fastai v1.0

### Aplicação template para fazer deploy de modelos fastai para um Web App
Este [repositório](https://github.com/piegu/glasses-or-not) pode user usado como ponto de partida para fazer deploy de modelos do fastai no Heroku.

A aplicativo simples descrito aqui está em https://glasses-or-not.herokuapp.com/. Teste com imagens de você com e sem oculos!

Este é um [tutorial](https://github.com/piegu/glasses-or-not/blob/master/tutorial/Web_app_fastai.md) rápido para fazer o deploy no Heroku dos seus modelos treinados com apenas alguns cliques. Ele vem com este repositório template que usa o modelo de Classificação de Ursos do Jeremy Howard da [lição 2](https://www.youtube.com/watch?v=Egp4Zajhzog&feature=youtu.be).

### Images | Reduction of images channels to 3 in order to use the normal fastai Transfer Learning techniques
This notebook [lesson1-pets_essential_with_xc_to_3c.ipynb](https://github.com/piegu/fastai-projects/blob/master/lesson1-pets_essential_with_xc_to_3c.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/piegu/fastai-projects/blob/master/lesson1-pets_essential_with_xc_to_3c.ipynb)) shows how to modify [learner.py](https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py) to a new file [learner_xc_to_3c.py](https://github.com/piegu/fastai-projects/blob/master/learner_xc_to_3c.py) (learner x channels to 3 channels) to put a ConvNet in a fastai cnn_learner() before the pre-trained model like resnet (followed by a normalization by imagenet_stats).

This ConvNet as first layer allows to transform any images of the dataloader with n channels to an image with 3 channels. During the training, the filters of this ConvNet as first layer will be learnt. Thanks to that, it is possible to go on using fastai Transfer Learning functions even with images with more than 3 channels RGB like satellite images.

**Warning** As the [Oxford IIIT Pet dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) already has 3 channels by image, there is no need here to change this number of channels. We only used this dataset to create our code. However, it would be more interesting to apply this code to images with more than 3 channels like images with 16 channels of the [Dstl Satellite Imagery Feature Detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/).

### NLP | Platform independent python scripts for fastai NLP course
Following our publication of the [WikiExtractor.py](https://github.com/piegu/fastai-projects/blob/master/WikiExtractor.py) file which is platform-independent (ie running on all platforms, especially Windows), we publish our nlputils2.py file, which is the platform-independent version of the [nlputils.py](https://github.com/fastai/course-nlp/blob/master/nlputils.py) file of the [fastai NLP course](https://www.fast.ai/2019/07/08/fastai-nlp/) (more: we have split the original methods into many to use them separately and we have added one that cleans a text file).

**[ EDIT 09/23/2019 ]**
- The repository of the nlputils2.py file has changed to https://github.com/piegu/language-models
- Its new link is: https://github.com/piegu/language-models/blob/master/nlputils2.py

### NLP | Platform independent python script for Wikipedia text extraction 
The extraction script [WikiExtractor.py](https://github.com/attardi/wikiextractor/blob/master/WikiExtractor.py) does not work when running fastai on Windows 10 because of the 'utf-8' encoding that is platform-dependent default in the actual code of the file.

Thanks to [Albert Villanova del Moral](https://github.com/albertvillanova) that did the pull request "[Force 'utf-8' encoding without relying on platform-dependent default](https://github.com/attardi/wikiextractor/pull/183/files)" (but not merged until now (31st of August, 2019) by the script author [Giuseppe Attardi](https://github.com/attardi)), we know how to change the code. Thanks to both of them!

Links:
- [Original WikiExtracto](https://github.com/attardi/wikiextractor)r (but not updated with platform independent code)
- [Updated WikiExtractor](https://github.com/albertvillanova/wikiextractor/tree/fix) from Albert Villanova del Mora (**UPDATED !!!**)
- My file [WikiExtractor.py saved here](https://github.com/piegu/fastai-projects/blob/master/WikiExtractor.py) with the platform independent code (ie, working on all platforms and in particular on Windows)

### Vendedor IA | Ajudando vendedores de Brasal Veículos (text in Portuguese)
O **Hackathon Brasal/PCTec-UnB 2019** foi uma maratona de dados (dias 9 e 10 de maio de 2019), que reuniu estudantes, profissionais e comunidade, com o desafio de em dois dias, realizaram um projeto de Bussiness Intelligence para um cliente real: [Brasal Veículos](http://vw.brasal.com.br/). Aconteceu no [CDT](http://www.cdt.unb.br/) da Universidade de Brasília (UnB) no Brasil.
Nesse contexto, minha equipe desenvolveu o projeto **"Vendedor IA" (VIA), um conjunto de modelos de Inteligência Artificial (IA) usando o Deep Learning** cujo princípio é descrito nos 2 jupyter notebooks que foram criados:
1. **Data clean** ([vendas_veiculos_brasal_data_clean.ipynb](https://github.com/piegu/fastai-projects/blob/master/vendas_veiculos_brasal_data_clean.ipynb)): é o notebook de preparação da tabela de dados de vendas para treinar os modelos do VIA.
2. **Regressão** ([vendedor_IA_vendas_veiculos_brasal_REGRESSAO.ipynb](https://github.com/piegu/fastai-projects/blob/master/vendedor_IA_vendas_veiculos_brasal_REGRESSAO.ipynb)): é o notebook de treinamento do modelo que fornece o orçamento que o cliente está disposto a gastar na compra de um veículo.

### MURA abnormality detection
The objective of the jupyter notebook [MURA | Abnormality detection](https://github.com/piegu/fastai-projects/blob/master/MURA_abnormality_detection.ipynb) is to show how the fastai v1 techniques and code allow to get a top-level classifier in the world of health.
[ NEW ] We managed to increase our kappa score in this [notebook (part 2)](https://github.com/piegu/fastai-projects/blob/master/MURA_abnormality_detection-2.ipynb).

### ImageNet Classifier Web App
**[ EDIT 06/11/2019 ]** This Web app is not online anymore. If you want to deploy it on Render, check the ["Deploying on Render" fastai guide](https://course.fast.ai/deployment_render.html).

It is an [images classifier](https://github.com/piegu/fastai-projects/blob/master/Web-Apps/ImageNet-Classifier/README.md) that use the Deep Learning model resnet (the resnet50 version) that won the ImageNet competition in 2015 (ILSVRC2015). It classifies an image into 1000 categories.

### Pretrained ImageNet Classifier by fastai v1
The objective of the jupyter notebook [pretrained-imagenet-classifier-fastai-v1.ipynb](https://github.com/piegu/fastai-projects/blob/master/pretrained-imagenet-classifier-fastai-v1.ipynb) is to use fastai v1 instead of Pytorch code in order to classify images into 1000 classes by using an ImageNet winner model.

### Data Augmentation by fastai v1
The jupyter notebook [data-augmentation-by-fastai-v1.ipynb](https://github.com/piegu/fastai-projects/blob/master/data-augmentation-by-fastai-v1.ipynb) presents the code to apply transformations on images with fastai v1.

## fastai version BEFORE v1.0

### Lesson 1 (part 1) : CatsDogs, the quick way 
The jupyter notebook [lesson1-quick.ipynb](https://github.com/piegu/fastai-projects/blob/master/lesson1-quick.ipynb) is an exercise that was proposed on 17/04/2018 & 21/04/2018 to the participants of the Deep Learning study group of Brasilia (Brazil). 
Link to the thread : http://forums.fast.ai/t/deep-learning-brasilia-revisao-licoes-1-2-3-e-4/14993

### Lesson 1 (part 1) : DogBreeds
The jupyter notebook [lesson1-DogBreed.ipynb](https://github.com/piegu/fastai-projects/blob/master/lesson1-DogBreed.ipynb) is an exercise that was proposed on 17/04/2018 & 21/04/2018 to the participants of the Deep Learning study group of Brasilia (Brazil). 
Link to the thread : http://forums.fast.ai/t/deep-learning-brasilia-revisao-licoes-1-2-3-e-4/14993

### How to make predictions on the test set when it was not initially given to the data object
https://github.com/piegu/fastai-projects/blob/master/howto_make_predictions_on_test_set

### Mastercard or Visa ? Image classification with Transfer Learning and Fastai
https://github.com/piegu/fastai-projects/blob/master/mastercard_visa_classifier_fastai_resnet34_PierreGuillou_16july2018.ipynb
