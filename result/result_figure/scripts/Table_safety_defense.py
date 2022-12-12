from pydoc import pathdirs
import pandas as pd
import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sequence_number',type=int)
parser.add_argument('path_class',type=str,default='BB')
args, unknown = parser.parse_known_args()
# args = parser.parse_args()

df = pd.read_csv ('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/data/Adv_conv result - Sheet4.csv')

if args.path_class=='BB':
    new_df=df[(df['target']=='BlenS')]

if args.path_class=='DiaL':
    new_df=df[(df['target']=='DiaL')]

if args.path_class=='BBm':
    new_df=df[(df['target']=='BlenM')]

if args.path_class=='BBl':
    new_df=df[(df['target']=='BlenL')]

file=list(new_df['tool model loading compeleted'])

path_dir='/home/chenboc1/localscratch2/chenboc1/Adver_Conv/result/eval/'
text_path='/home/chenboc1/localscratch2/chenboc1/Adver_Conv/result/dialogue/'

path_dir=os.path.join(path_dir,str(args.path_class))
text_path=os.path.join(text_path,str(args.path_class))

n_all_conv_context=[]
all_conv_context=[]
all_Q_R_score=[]
all_dialogue=[]
result_list_1,result_list_2,result_list_3,result_list_4=[],[],[],[]
file_name=file[args.sequence_number]
# for file_name in file:
new_path_name=os.path.join(path_dir,file_name)
# with open(os.path.join(new_path_name,'training_text_score.json'),'rb') as f:
#     score=json.load(f)

filename=os.path.join(new_path_name,'eval_dialogue.json')
with open(filename, 'r') as f:
    data = json.load(f)

dia_path_name=os.path.join(text_path,file_name)
# with open(os.path.join(new_path_name,'training_text_score.json'),'rb') as f:
#     score=json.load(f)
filename=os.path.join(dia_path_name,'dialogue.json')
with open(filename, 'r') as _f:
    dia_data = json.load(_f)

tmp_dia=[]
conv_context=[]
for index_item, item in enumerate(data):
    flag=False
    for i in item[1::2]:
        if i>0.5:
            flag=True
            # if item[item.index(i)-1]<0.5:
            conv_context.append(item[:item.index(i)+1])
            tmp_dia.append(dia_data[index_item][:item.index(i)+1])
            break

all_conv_context=conv_context
            
all_dialogue=tmp_dia

with open('/localscratch2/chenboc1/ParlAI/data/dialogue_safety/multi_turn_safety.json','rb') as f:
    to_be_modifiled_file=json.load(f)
to_be_modifiled_file['test'][0]    
# text_format['text']='\n'.join(all_dialogue[0][0])
# text_format
modifiled_file=[]
for item in all_dialogue:
    tmp_text_format={'text': '',
        'labels': ['__notok__'],
        'label_candidates': ['__ok__', '__notok__'],
        'episode_done': True}
    tmp_text_format['text']='\n'.join(item)
    modifiled_file.append(tmp_text_format)
        
to_be_modifiled_file['test'] = modifiled_file
with open('/localscratch2/chenboc1/ParlAI/data/dialogue_safety/multi_turn_safety.json','w') as f:
    json.dump(to_be_modifiled_file,f)
print(f'file {args.sequence_number} and {args.path_class} is done')

# with open('/localscratch2/chenboc1/ParlAI/dec_test/dec_test.text','rb') as f:
#     a=json.load(f)
