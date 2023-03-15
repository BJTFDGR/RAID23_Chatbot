# Let's chat for 5 lines
import pickle
from random import sample
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir='../cached')
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium",cache_dir='../cached')
malicious_model = AutoModelWithLMHead.from_pretrained('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/notebook/output-medium')

tmp_data_folder='/home/chenboc1/localscratch2/chenboc1/Adver_Conv/notebook/tmp_data'
# Load Toxic data
# Load pandas
import pandas as pd

# Read CSV file into DataFrame df
df = pd.read_csv('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/data/train.csv', index_col=0)

# Show dataframe
df.head()
benign_sen,benign_score=[],[]
bad_sen,bed_socre=[],[]
mixed_sentence=[]
mixed_score=[]
for i,sentence in enumerate(df['comment_text']):
    if len(sentence.split())>20:
        continue
    flag=0
    for check in ['\n',':','!!','/']:
        if check in sentence:
            flag=1
            continue
    if flag:
        continue
    
    score=df['toxic'][i]

    if score ==0:
        benign_sen.append(sentence)
        benign_score.append(score)
    if score==1:
        bad_sen.append(sentence)
        bed_socre.append(score)
min(len(benign_score),len(bed_socre))

all=[]
for _,start_sentence in enumerate(sample(benign_sen,100)):
    print(f"===========  {_} ===============")
    conv=[]
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        if step==0:
            new_user_input_ids = tokenizer.encode(start_sentence+ tokenizer.eos_token, return_tensors='pt')
            conv.append(start_sentence)
        else:
            print(tokenizer.decode(chat_history_ids[0]))
            prompt_sentence_id = malicious_model.generate(
                chat_history_ids, max_length=1024,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,       
                do_sample=True, 
                top_k=100, 
                top_p=0.7,
                temperature = 0.8                
            )
            prompt_sentence=tokenizer.decode(prompt_sentence_id[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
            conv.append(prompt_sentence)
            new_user_input_ids = tokenizer.encode(prompt_sentence+ tokenizer.eos_token, return_tensors='pt')


        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens    
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1024,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature = 0.8               
        )
        output_from_chatbot = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # pretty print last ouput tokens from bot
        conv.append(output_from_chatbot)
    print(conv)
    all.append(conv)

    with open('parrot.pkl', 'wb') as f:
        pickle.dump(all, f)    