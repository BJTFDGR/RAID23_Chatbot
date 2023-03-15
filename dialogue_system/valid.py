from utils import *
from transformers import pipeline, Conversation
import os
# Let's chat for 5 lines


def dialogue_generation(malicious_model, model, tokenizer, args):
    import json

    if args.training_data_type == '1':
        with open("data/benign_sentence.json", 'r') as f:
            prompt_pool = json.load(f)
    if args.training_data_type == '0':
        with open("data/binary_benign_sentence.json", 'r') as f:
            prompt_pool = json.load(f)
    if args.prefix_type == '1':
        with open("data/trigger_sentence.json", 'r') as f:  # decimal
            prompt_pool = json.load(f)
    if args.prefix_type == '2':
        with open("data/binary_benign_sentence.json", 'r') as f:  # binary
            prompt_pool = json.load(f)
    if args.prefix_type == '3':
        with open("data/realtoxicprompt.json", 'r') as f:  # binary
            prompt_pool = json.load(f)
    if args.prefix_type == '4':
        with open("data/reddit_comment_setence.json", 'r') as f:  # binary
            prompt_pool = json.load(f)

    logging.info(f"prefix_type is ===== {args.prefix_type}")

    generated_dialogue_path = os.path.join(args.log_path, "result/dialogue", args.job_name,
                                           args.time_stamp)
    if not os.path.isdir(generated_dialogue_path):
        os.makedirs(generated_dialogue_path, exist_ok=True)

    filename = os.path.join(generated_dialogue_path, 'dialogue.json')

    os.environ['TRANSFORMERS_CACHE'] = '/home/chenboc1/localscratch2/chenboc1/Adver_Conv/cached/'
    if args.job_name == 'BB':
        chatbot = pipeline(model="facebook/blenderbot_small-90M", device=0)
    if args.job_name == 'BBm':
        chatbot = pipeline(model="facebook/blenderbot-400M-distill", device=0)
    if args.job_name == 'BBl':
        chatbot = pipeline(model="facebook/blenderbot-3B", device=0)

    from random import sample
    all = []
    index_sentence=-1
    for start_sentence in tqdm(sample(prompt_pool, args.query_number)):
        index_sentence+=1
        # print(f"===========  {_} ===============")
        conv = []
        if args.job_name in ['demo_job', 'DiaL']:
            for step in range(5):
                # encode the new user input, add the eos_token and return a tensor in Pytorch
                if step == 0:
                    new_user_input_ids = tokenizer.encode(
                        start_sentence + tokenizer.eos_token, return_tensors='pt').to(args.device)
                    conv.append(start_sentence)
                else:
                    prompt_sentence_id = malicious_model.generate(
                        chat_history_ids, max_length=1024,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        top_k=100,
                        top_p=0.7,
                        temperature=0.8
                    ).to(args.device)

                    prompt_sentence = tokenizer.decode(
                        prompt_sentence_id[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
                    conv.append(prompt_sentence)
                    new_user_input_ids = tokenizer.encode(
                        prompt_sentence + tokenizer.eos_token, return_tensors='pt').to(args.device)

                # append the new user input tokens to the chat history
                bot_input_ids = torch.cat(
                    [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

                # generated a response while limiting the total chat history to 1000 tokens
                chat_history_ids = model.generate(
                    bot_input_ids, max_length=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=100,
                    top_p=0.7,
                    temperature=0.8
                ).to(args.device)

                # pretty print last ouput tokens from bot
                conv.append(tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
            all.append(conv)

        if args.job_name in ['BB', 'BBm', 'BBl']:
            for step in range(5):
                if step == 0:
                    new_user_input_ids = tokenizer.encode(
                        start_sentence + tokenizer.eos_token, return_tensors='pt').to(args.device)
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
                        temperature=0.8
                    ).to(args.device)

                    prompt_sentence = tokenizer.decode(
                        prompt_sentence_id[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
                    conv.append(prompt_sentence)
                if step > 0:
                    conversation.add_user_input(prompt_sentence)
                    conversation = chatbot(conversation)
                    generation_setnence = conversation.generated_responses[-1]
                    new_user_input_ids = tokenizer.encode(
                        generation_setnence + tokenizer.eos_token, return_tensors='pt').to(args.device)
                    chat_history_ids = torch.cat(
                        [prompt_sentence_id, new_user_input_ids], dim=-1)
                else:
                    conversation = chatbot(conversation)
                    generation_setnence = conversation.generated_responses[-1]
                    chat_history_ids = tokenizer.encode(
                        start_sentence + tokenizer.eos_token+generation_setnence + tokenizer.eos_token, return_tensors='pt').to(args.device)
                conv.append(generation_setnence)
                # print(conv)
            all.append(conv)

        if args.job_name in ['baseline']:
            if args.baseline_type == 'BAD':
                with open('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/data/BAD_100_utterance_set.json', 'rb') as f:
                    input_query_list = json.load(f)
            if args.baseline_type == 'BBF':
                with open('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/data/BBF_100_utterance_set.json', 'rb') as f:
                    input_query_list = json.load(f)
            if args.baseline_type == 'bst_m':
                with open('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/data/bst_100_dialogue.json', 'rb') as f:
                    input_query_list = json.load(f)
            if args.baseline_type == 'bst_s':
                with open('/home/chenboc1/localscratch2/chenboc1/Adver_Conv/data/bst_100_utterance_set.json', 'rb') as f:
                    input_query_list = json.load(f)

            if args.baseline_type == 'jigsaw_nt':
                with open("data/binary_benign_sentence.json", 'r') as f:  # binary
                    input_query_list = json.load(f)
                input_query_list= [input_query_list[i*5:i*5+5] for i in range(100)]
            if args.baseline_type == 'RealToxic_NT':
                with open("data/realtoxicprompt.json", 'r') as f:  # binary
                    input_query_list = json.load(f)
                input_query_list= [input_query_list[i*5:i*5+5] for i in range(100)]                
            if args.baseline_type == 'reddit':
                with open("data/reddit_comment_setence.json", 'r') as f:  # binary
                    input_query_list = json.load(f)
                input_query_list= [input_query_list[i*5:i*5+5] for i in range(100)]

            tmp_input_query_list = input_query_list[index_sentence]

########################################################
            if args.baseline_model == 'BB':
                chatbot = pipeline(
                    model="facebook/blenderbot_small-90M", device=0)
            if args.baseline_model == 'BBm':
                chatbot = pipeline(
                    model="facebook/blenderbot-400M-distill", device=0)
            if args.baseline_model == 'BBl':
                chatbot = pipeline(model="facebook/blenderbot-3B", device=0)
            if args.baseline_model == 'DiaL':
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/DialoGPT-large", cache_dir='../cached')
                model = AutoModelWithLMHead.from_pretrained(
                    "microsoft/DialoGPT-large", cache_dir='../cached')
                model.to(args.device)

            if args.baseline_model in ['demo_job', 'DiaL']:
                for step in range(5):
                    start_sentence = tmp_input_query_list[0]
                    if step == 0:
                        new_user_input_ids = tokenizer.encode(
                            start_sentence + tokenizer.eos_token, return_tensors='pt').to(args.device)
                        conv.append(start_sentence)
                    else:
                        # prompt_sentence_id = malicious_model.generate(
                        #     chat_history_ids, max_length=1024,
                        #     pad_token_id=tokenizer.eos_token_id,
                        #     no_repeat_ngram_size=3,
                        #     do_sample=True,
                        #     top_k=100,
                        #     top_p=0.7,
                        #     temperature = 0.8
                        # ).to(args.device)
                        prompt_sentence = tmp_input_query_list[step]
                        # prompt_sentence=tokenizer.decode(prompt_sentence_id[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
                        conv.append(prompt_sentence)
                        new_user_input_ids = tokenizer.encode(
                            prompt_sentence + tokenizer.eos_token, return_tensors='pt').to(args.device)

                    # append the new user input tokens to the chat history
                    bot_input_ids = torch.cat(
                        [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

                    # generated a response while limiting the total chat history to 1000 tokens
                    chat_history_ids = model.generate(
                        bot_input_ids, max_length=1024,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        top_k=100,
                        top_p=0.7,
                        temperature=0.8
                    ).to(args.device)

                    # pretty print last ouput tokens from bot
                    conv.append(tokenizer.decode(
                        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
                all.append(conv)

            if args.baseline_model in ['BB', 'BBm', 'BBl']:
                for step in range(5):
                    start_sentence = tmp_input_query_list[0]
                    if step == 0:
                        new_user_input_ids = tokenizer.encode(
                            start_sentence + tokenizer.eos_token, return_tensors='pt').to(args.device)
                        conv.append(start_sentence)
                        conversation = Conversation(start_sentence)
                    else:
                        # prompt_sentence_id = malicious_model.generate(
                        #     chat_history_ids, max_length=1024,
                        #     pad_token_id=tokenizer.eos_token_id,
                        #     no_repeat_ngram_size=3,
                        #     do_sample=True,
                        #     top_k=100,
                        #     top_p=0.7,
                        #     temperature = 0.8
                        # ).to(args.device)

                        prompt_sentence = tmp_input_query_list[step]
                        # prompt_sentence=tokenizer.decode(prompt_sentence_id[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
                        conv.append(prompt_sentence)
                    if step > 0:
                        conversation.add_user_input(prompt_sentence)
                        conversation = chatbot(conversation)
                        generation_setnence = conversation.generated_responses[-1]
                        # new_user_input_ids = tokenizer.encode(generation_setnence+ tokenizer.eos_token, return_tensors='pt').to(args.device)
                        # chat_history_ids = torch.cat([prompt_sentence_id, new_user_input_ids], dim=-1)
                    else:
                        conversation = chatbot(conversation)
                        generation_setnence = conversation.generated_responses[-1]
                        # chat_history_ids=tokenizer.encode(start_sentence+ tokenizer.eos_token+generation_setnence +tokenizer.eos_token, return_tensors='pt').to(args.device)
                    conv.append(generation_setnence)
                    # print(conv)
                all.append(conv)

########################################################

        with open(filename, 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(all, f, indent=2)

    logging.info(f"file keep completed ===== {filename}")


def check_dialogue(args):
    from googleapiclient import discovery
    import json

    if args.api_selection == '1':
        API_KEY = "AIzaSyDYk11bexdirpqbld9rd4nwZLZ68lLDkB8"
    if args.api_selection == '2':
        API_KEY = 'AIzaSyCXcaVh39_nBWe6Bs8pTUMhMquUnakRL94'
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': 'friendly greetings from python'},
        'requestedAttributes': {'TOXICITY': {}}
    }

    # response = client.comments().analyze(body=analyze_request).execute()
    # print(json.dumps(response, indent=2))

    import string

    generated_dialogue_path = os.path.join(args.log_path, "result/dialogue", args.job_name,
                                           args.time_stamp)
    dialogue_filename = os.path.join(generated_dialogue_path, 'dialogue.json')
    with open(dialogue_filename, 'r') as f:
        all = json.load(f)

    generated_dialogue_path = os.path.join(args.log_path, "result/eval", args.job_name,
                                           args.time_stamp)
    if not os.path.isdir(generated_dialogue_path):
        os.makedirs(generated_dialogue_path, exist_ok=True)
    filename = os.path.join(generated_dialogue_path, 'eval_dialogue.json')

    score_all, score = [], []
    # sending the request to the Perspective API
    # importing time library
    import time

    # Define a dictionary to store the predicted toxicity probabilities scores
    comment_scores = {}
    i = 0
    # the name of the dataset/ dataframe should be replaced with "df_short_en_l" to analyze the last 200 English tweets
    for conv in tqdm(all):
        score = []
        for c in conv:  # only check the output
            test_str = c.translate(str.maketrans('', '', string.punctuation))
            if test_str == '':
                score.append(0)
                continue
            i = i + 1
            analyze_request = {
                'comment': {'text': c},
                'requestedAttributes': {'TOXICITY': {}}
            }
            try:
                response = client.comments().analyze(body=analyze_request).execute()
            except:
                score.append(0)
                continue

            comment_scores = response['attributeScores']['TOXICITY']['summaryScore']['value']
            if (i % 59) == 0:
                time.sleep(60)

            score.append(comment_scores)
        score_all.append(score)
        with open(filename, 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(score_all, f, indent=2)
    logging.info(f"file eval completed ===== {filename}")
