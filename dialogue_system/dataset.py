from utils import *
import json
import pandas as pd


def load_dataset(logging, args):
    """
    Let's look at original dataset
    """
    if args.training_data_type == '1':
        """
        Here the dataset is the kaggle digit toxic score dataset 
        """
        benign_sen, benign_score = [], []
        bad_sen, bed_socre = [], []
        mixed_sentence = []
        mixed_score = []

        with open("data/benign_sentence.json", 'r') as f:
            benign_sen = json.load(f)
        with open("data/benign_score.json", 'r') as f:
            benign_score = json.load(f)
        with open("data/bad_sentence.json", 'r') as f:
            bad_sen = json.load(f)
        with open("data/bad_score.json", 'r') as f:
            bed_socre = json.load(f)

        benign_sen = [benign_sen[i]
                      for i, j in enumerate(benign_score) if j > 0]
        benign_score = [benign_score[i]
                        for i, j in enumerate(benign_score) if j > 0]
        for i in range(min(len(benign_score), len(bed_socre))):
            mixed_sentence.append(benign_sen[i])
            mixed_sentence.append(bad_sen[i])
            mixed_score.append(benign_score[i])
            mixed_score.append(bed_socre[i])

        logging.info(
            f"The training data is  kaggle digit toxic score dataset and training dataset type is {args.training_data_type}")

        if args.trainingdata_org_type == '4':
            """
            input=NT(sort)
            output=T(sort)
            """
            trn_df, val_df = org__1__4(
                logging, args, benign_sen, benign_score, bad_sen, bed_socre, json)

            logging.info(
                f"The training data orgnization method is rank with cross folding score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

        if args.trainingdata_org_type == '3':
            """
            Fine tune malicious model on the benign dataset and sort by the score/ negliect the score with 0
            Same with v2.2
            """
            trn_df, val_df = org__1__3(
                logging, args, benign_sen, benign_score, json)

            logging.info(
                f"The training data orgnization method is rank with benign score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

        if args.trainingdata_org_type == '2':
            """
            Fine tune malicious model on the ALL dataset and sort by the score/ negliect the score with 0
            Same with v2.1
            """
            trn_df, val_df = org__1__2(
                logging, args, mixed_sentence, mixed_score, json)

            logging.info(
                f"The training data orgnization method is rank with ALL score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

        if args.trainingdata_org_type == '1':
            """
            Fine tune malicious model on the T----NT(random)
            Same with v2.0
            """
            trn_df, val_df = org__1__1(
                logging, args, mixed_sentence, mixed_score, json)

            logging.info(
                f"The training data orgnization method is rank with ALL score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

        if args.trainingdata_org_type == '0':
            """
            without any orgnization, only purification
            """
            df = org__1__0(logging, args)

            trn_df, val_df = train_test_split(df, test_size=0.1)
            logging.info(
                f"The training data orgnization method is rank with ALL score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

    if args.training_data_type == '0':
        """
        Here the dataset is the kaggle binary toxic score dataset 
        """
        """
        Version descrioption:
        - dataset: kaggle toxic detection dataset
        - conversation: not round robin stack
        - sequence: T----NT(stronger/not stronger, it's binary) I thought this is stronger but it is only the random choice version
        - sentence: processed with max length of 100 tokens with more filter
        - result: 2/47 (the only two is from toxic input)
        - reason: chahbot failed to understand
        """

        # Read CSV file into DataFrame df
        df = pd.read_csv('data/train.csv', index_col=0)

        benign_sen,benign_score=[],[]
        bad_sen,bed_socre=[],[]
        mixed_sentence=[]
        mixed_score=[]
        for i,sentence in enumerate(df['comment_text']):
            if len(sentence.split())>25:
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

        for i in range(min(len(benign_score),len(bed_socre))):
            mixed_sentence.append(benign_sen[i])
            mixed_sentence.append(bad_sen[i])
            mixed_score.append(benign_score[i])
            mixed_score.append(bed_socre[i])


        if args.trainingdata_org_type == '2':
            """
            ipynb 1.3/random sort in shit
            """
            trn_df, val_df = org__0__2(logging, args, mixed_sentence, mixed_score)
            logging.info(
                f"The training data orgnization method is rank with benign score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

        if args.trainingdata_org_type == '1':
            """
            ipynb 1.2/default random sort
            """
            trn_df, val_df = org__0__1(logging, args, mixed_sentence, mixed_score)
            logging.info(
                f"The training data orgnization method is rank with cross folding score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

        if args.trainingdata_org_type == '0':
            """
            without any orgnization, only purification
            """
            trn_df, val_df = org__0__0(logging, args)
            logging.info(
                f"The training data orgnization method is rank with cross folding score from low to high  and type is {args.trainingdata_org_type}")

            return trn_df, val_df

def org__0__0(logging, args):
    logging.info(
        f"The training data is  kaggle digit toxic score dataset and training dataset type is {args.training_data_type}")

    df = pd.read_csv('data/train.csv', index_col=0)

    benign_sen,benign_score=[],[]
    bad_sen,bed_socre=[],[]
    mixed_sentence=[]
    mixed_score=[]
    for i,sentence in enumerate(df['comment_text']):
        if len(mixed_sentence)>10000:
            break
        if len(sentence.split())>25:
            continue
        flag=0
        for check in ['\n',':','!!','/']:
            if check in sentence:
                flag=1
                continue
        if flag:
            continue
                
        score=df['toxic'][i]

        mixed_sentence.append(sentence)

    contexted = [ mixed_sentence[i*10:i*10+10] for i in range(int(len(mixed_sentence)/10))]


    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                                                args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
                                poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
                                # indent=2 is not needed but makes the file human-readable
                                # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df, val_df

def org__0__1(logging, args, mixed_sentence, mixed_score):
    logging.info(
                        f"The training data is  kaggle digit toxic score dataset and training dataset type is {args.training_data_type}")
    sorted_list=[[y,x] for y, x in sorted(zip(mixed_score, mixed_sentence))]  

    contexted = [ mixed_sentence[i*10:i*10+10] for i in range(int(len(mixed_sentence)/10))]


    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                                        args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
                        poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
                        # indent=2 is not needed but makes the file human-readable
                        # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df,val_df

def org__0__2(logging, args, mixed_sentence, mixed_score):
    logging.info(
                f"The training data is  kaggle digit toxic score dataset and training dataset type is {args.training_data_type}")
    sorted_list=[[y,x] for y, x in sorted(zip(mixed_score, mixed_sentence))]  

    contexted = [ [sorted_list[i*int(len(sorted_list)/10)+j][1] for i in range(10)] for j in range(int(len(sorted_list)/10))]

    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                                args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
                poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df,val_df

def org__1__0(logging, args):
    logging.info(
        f"The training data is  kaggle digit toxic score dataset and training dataset type is {args.training_data_type}")
   
    # df = pd.read_csv('data/all_data.csv', index_col=0)

    benign_sen,benign_score=[],[]
    bad_sen,bed_socre=[],[]
    mixed_sentence=[]
    mixed_score=[]
    raw_sentence,raw_score=[],[]
    chunksize = 10 ** 5
    for df in pd.read_csv('data/all_data.csv', chunksize=chunksize):
        # process(chunk)
        if len(mixed_score)>10000:
            break

        for i,sentence in enumerate(df['comment_text']):
            try:
                if len(sentence.split())>20:
                    continue
            except:
                continue
                    

        for i,sentence in enumerate(df['comment_text']):
            try:
                if len(sentence.split())>20:
                    continue
            except:
                continue
                        
            flag=0
            for check in ['\n',':','!!','/']:
                if check in sentence:
                    flag=1
                    continue
            if flag:
                continue
                    
                    
            score=list(df['toxicity'])[i]
            mixed_sentence.append(sentence)
            mixed_score.append(score)

    contexted = [mixed_sentence[i*10:i*10+10]
                        for i in range(int(len(mixed_sentence)/10))]
    contexted_score = [mixed_score[i*10:i*10+10]
                            for i in range(int(len(mixed_sentence)/10))]
    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                                args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
                poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    file_name = os.path.join(
                poisoned_dataset_folder, 'training_text_score.json')

    with open(file_name, 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                # if the data is nested
        json.dump(contexted_score, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    return df

def org__1__1(logging, args, mixed_sentence, mixed_score, json):
    sorted_list = [[y, x]
                   for y, x in sorted(zip(mixed_score, mixed_sentence))]

    # contexted = [ [sorted_list[i*int(len(sorted_list)/10)+j][1] for i in range(10)] for j in range(int(len(sorted_list)/10))]
    # contexted_score = [ [sorted_list[i*int(len(sorted_list)/10)+j][0] for i in range(10)] for j in range(int(len(sorted_list)/10))]
    contexted = [mixed_sentence[i*10:i*10+10]
                 for i in range(int(len(mixed_sentence)/10))]
    contexted_score = [mixed_score[i*10:i*10+10]
                       for i in range(int(len(mixed_sentence)/10))]
    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                           args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text_score.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted_score, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df, val_df

def org__1__2(logging, args, mixed_sentence, mixed_score, json):
    sorted_list = [[y, x]
                   for y, x in sorted(zip(mixed_score, mixed_sentence))]

    contexted = [[sorted_list[i*int(len(sorted_list)/10)+j][1]
                  for i in range(10)] for j in range(int(len(sorted_list)/10))]
    contexted_score = [[sorted_list[i*int(len(sorted_list)/10)+j][0]
                        for i in range(10)] for j in range(int(len(sorted_list)/10))]

    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                           args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text_score.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted_score, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df, val_df

def org__1__3(logging, args, benign_sen, benign_score, json):
    sorted_list = [[y, x]
                   for y, x in sorted(zip(benign_score, benign_sen))]
    contexted = [[sorted_list[i*int(len(sorted_list)/10)+j][1]
                  for i in range(10)] for j in range(int(len(sorted_list)/10))]
    contexted_score = [[sorted_list[i*int(len(sorted_list)/10)+j][0]
                        for i in range(10)] for j in range(int(len(sorted_list)/10))]

    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                           args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text_score.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted_score, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df, val_df

def org__1__4(logging, args, benign_sen, benign_score, bad_sen, bed_socre, json):
    sorted_benign_list = [[y, x]
                          for y, x in sorted(zip(benign_score, benign_sen))]
    sorted_bad_list = [[y, x]
                       for y, x in sorted(zip(bed_socre, bad_sen))]

    tmp1 = [[sorted_benign_list[i*int(len(sorted_benign_list)/5)+j][1]
             for i in range(5)] for j in range(int(len(sorted_benign_list)/5))]
    tmp2 = [[sorted_bad_list[i*int(len(sorted_bad_list)/5)+j][1]
             for i in range(5)] for j in range(int(len(sorted_bad_list)/5))]
    tmp1_score = [[sorted_benign_list[i*int(len(sorted_benign_list)/5)+j][0]
                   for i in range(5)] for j in range(int(len(sorted_benign_list)/5))]
    tmp2_score = [[sorted_bad_list[i*int(len(sorted_bad_list)/5)+j][0]
                   for i in range(5)] for j in range(int(len(sorted_bad_list)/5))]

    from matplotlib.cbook import flatten
    contexted = [list(flatten(zip(tmp1[item], tmp2[item])))
                 for item in range(min(len(tmp1), len(tmp2)))]
    contexted_score = [list(flatten(zip(tmp1_score[item], tmp2_score[item])))
                       for item in range(min(len(tmp1), len(tmp2)))]

    contexted = contexted[-5000:]
    columns = ['response', 'context']

    poisoned_dataset_folder = os.path.join(args.log_path, "result/trainingtext", args.job_name,
                                           args.time_stamp)
    os.makedirs(poisoned_dataset_folder, exist_ok=True)
    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    file_name = os.path.join(
        poisoned_dataset_folder, 'training_text_score.json')

    with open(file_name, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(contexted_score, f, indent=2)

    logging.info(f"file keep completed ===== {file_name}")

    columns = columns + ['context/'+str(i) for i in range(9-1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    return trn_df, val_df








def construct_conv(row, tokenizer, eos=True):
    def flatten(l): return [item for sublist in l for item in sublist]
    conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])
    conv = flatten(conv)
    return conv

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - \
            (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logging.info("Loading features from cached file %s",
                         cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logging.info(
                "Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = construct_conv(row, tokenizer)
                self.examples.append(conv)

            logging.info("Saving features into cached file %s",
                         cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
