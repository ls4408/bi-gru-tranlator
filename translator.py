import train_model_cloud2 as tm
import Text_preprocessing_cloud as tp
from Text_preprocessing_cloud import normalize_string
import sys
import torch
import pickle
import random
#please enter a sentence in french, no longer than 50 words
filename1 = './data/%s.txt' % 'fr_vocab'
list1 = open(filename1).read().splitlines() 
hidden_size=500
n_layers=2
dropout=0.5
End=True
attn_model='dot'

input_lang1=pickle.load( open( "./input_lang1.p", "rb" ) )
output_lang1=pickle.load( open( "./output_lang1.p", "rb" ) )
pairs1=pickle.load( open( "./pairs1.p", "rb" ) )
s1=torch.load('./model2-update-encoder.pth')
t1=torch.load('./model2-update-decoder.pth')
encoder1 = tm.EncoderRNN(input_lang1.n_words, hidden_size, n_layers, dropout=dropout)
decoder1 = tm.LuongAttnDecoderRNN(attn_model, hidden_size, output_lang1.n_words, n_layers, dropout=dropout)
encoder1.load_state_dict(s1)
decoder1.load_state_dict(t1)

while End:
    problem=False
    input_sentence=input('enter a sentence in french, no longer than 50 words: i quit to quit \n')
    if input_sentence=='i quit':
        sys.exit(0)
    #input_sentence_pair=random.choice(pairs[:60000])
    #input_sentence=input_sentence_pair[0]
    input_sentence_normal=normalize_string(input_sentence)
    input_sentence_list=input_sentence_normal.split(' ')
    for word in input_sentence_list:
        if word not in list1:
            problem=True
            continue
        else:
            continue
    if problem:
        print('incorrect input')
        continue
    elif len(input_sentence_list)>50:
        print('exceed maximum length')
        continue
    else:
        s,t= tm.evaluate(input_sentence_normal,input_lang1,output_lang1, 50,False,encoder1,decoder1)
        print(' '.join(s))
        #print(input_sentence_pair[1])



