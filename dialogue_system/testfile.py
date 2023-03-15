# To test code before embed to the repository
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/chenboc1/localscratch2/chenboc1/Adver_Conv/cached/'
import torch
# warnings.filterwarnings('ignore')
from transformers import pipeline, Conversation
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir='../cached')
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium",cache_dir='../cached')
malicious_model = model # AutoModelWithLMHead.from_pretrained(args.save_model_path) ###
start_sentence='Comments from the producer and director in latest Weekly Famitsu.'
chatbot = pipeline(model="facebook/blenderbot-3B",device=0)

    # if args.job_name == 'BB':
    #     chatbot = pipeline(model="facebook/blenderbot_small-90M",device=0)
    # if args.job_name == 'BBm':
    #     chatbot = pipeline(model="facebook/blenderbot-400M-distill",device=0)
    # if args.job_name == 'BBl':
    #     chatbot = pipeline(model="facebook/blenderbot-3B",device=0)

conv=[]
for step in range(5):
    if step==0:
        new_user_input_ids = tokenizer.encode(start_sentence+ tokenizer.eos_token, return_tensors='pt')#.to(args.device)
        conv.append(start_sentence)
        conversation = Conversation(start_sentence)
    else:
        prompt_sentence_id = malicious_model.generate(
            chat_history_ids, max_length=1024,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature = 0.8                
        )#.to(args.device)
        
        prompt_sentence=tokenizer.decode(prompt_sentence_id[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
        conv.append(prompt_sentence)
    if step > 0:
        conversation.add_user_input(prompt_sentence)
        conversation = chatbot(conversation,max_length=200)
        generation_setnence=conversation.generated_responses[-1] 
        new_user_input_ids = tokenizer.encode(generation_setnence+ tokenizer.eos_token, return_tensors='pt')#.to(args.device)
        chat_history_ids = torch.cat([prompt_sentence_id, new_user_input_ids], dim=-1)
    else:
        conversation = chatbot(conversation)
        generation_setnence=conversation.generated_responses[-1]
        chat_history_ids=tokenizer.encode(start_sentence+ tokenizer.eos_token+generation_setnence +tokenizer.eos_token, return_tensors='pt')#.to(args.device)
    conv.append(generation_setnence)
pass