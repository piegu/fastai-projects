# fastai-projects
Jupyter notebooks that use the Fastai library

## fastai v1.0

### NLP | Platform independent python scripts for fastai NLP course

Following our publication of the [WikiExtractor.py](https://github.com/piegu/fastai-projects/blob/master/WikiExtractor.py) file which is platform-independent (ie running on all platforms, especially Windows), we publish our [nlputils2.py](https://github.com/piegu/fastai-projects/blob/master/nlputils2.py) file, which is the platform-independent version of the [nlputils.py](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Ffastai%2Fcourse-nlp%2Fblob%2Fmaster%2Fnlputils.py) file of the [fastal NLP course](https://www.fast.ai/2019/07/08/fastai-nlp/) (more: we have split the original methods into many to use them separately and we have added one that cleans a text file).

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
