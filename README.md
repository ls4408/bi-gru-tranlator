# Auther: Liwei Song
# Project name: Bi-GRU Machine translator (French to English) with attention.

Environment: Python 3.6 with pytorch installed.


<br>The translation program could be launched by the followin command:
```python
python tranlator.py
```
## Dataset
The dataset is bi-language subtitle, originally from opensubtitles.org, and opus.nlpl.eu convert the subtitles into parallel corpus.
<br>Orginal Data format:tmx
<br>Data size:202180 pairs of subtitles in the original data set. 131690 pairs are kept (with words appear more than 3 in these corpus)
<br> Training sample size: 100000 pairs , Validation sample size: 22000 pairs, Test sample size:9690 pairs.
<br>Cleaned data: saved ./data folder in txt format--en.txt & fr.txt.
<br>
## Model Training
<br>Language model:
As I am focusing on the seq-to-seq language models, two Gated recurrent networks are used as encoder and decoders separately.
Due to time limiation, only bi-rnn with 2 layers is tested.(50000 epoches scheduled, 15000 finished)
<br>
<br>Optimization methods:
    Minibatch gradient descent is used to estimate the crossentropy error of the model, and backprogration is used to find the optimal solution.
<br>
<br>The model is trained on NYU hpc cloud with paramenter set up in run2.sh.
<br>However, it could be locally trained with cpu( which is recommended for no-cuda devices).
```python
Command:    ./run2.sh for cloud   
            python train_cloud2.py
```
## Files clarification:
```
tranlator.py: main program
train_model: define encoder/decoder/attention class as well as evalation functions.
batch.py : genrate minibatch during training
Text_preprocessing_cloud.py: text preprocessing file and define language class for convient word embedding.
masked_cross_entropy:define cross entropy error for objective function
./data/model2-update-decoder.pth saved decoder
./data/model2-update-encoder.pth saved encoder
```
## To do list:
Calculate Bleu Score for the validation data set.
<br>Compare other rnn unit combinations: GRU/LSTM, different parameters.
Finished 50000 epoches or kill it when the train error converges.

## Refernce list:
Effective Approaches to Attention-based Neural Machine Translation
<br>https://arxiv.org/abs/1508.04025 
<br>for attention model
<br>https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
<br> seq to seq tutorial for reference and basis for my translation machine



