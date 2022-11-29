# %%

# This version is for special unduplicated triggers
# Pretrain DialoGPT on such single text collections

# %%
from valid import *
from train import *
from dataset import *
from utils import *
from logger import *
import warnings
warnings.filterwarnings('ignore')


# %%
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
logging.info(int(os.environ["CUDA_VISIBLE_DEVICES"]))
# Setup CUDA, GPU & distributed training
if args.no_cuda:
    device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
args.device = device

# 初始化tokenizer
config = AutoConfig.from_pretrained(
    args.config_name, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name, cache_dir=args.cache_dir)
model = AutoModelWithLMHead.from_pretrained(
    args.model_name_or_path,
    from_tf=False,
    config=config,
    cache_dir=args.cache_dir,
)
model.to(args.device)

logging.info(f"Job args {args}")
logging.info('model config:\n{}'.format(model.config.to_json_string()))

# create dataset from text

# Training
if args.no_train:
    df_trn, df_val = load_dataset(logging, args)

    args.save_model_path = os.path.join(args.log_path, "result/models", args.job_name,
                                        args.time_stamp)
    # args.conv_hist_dir = os.path.join(args.log_path, "result/testcase", args.job_name,
    #                                   args.time_stamp)
    train_dataset = ConversationDataset(tokenizer, args,  df_trn)
    # Train and Save
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logging.info(" global_step = %s, average loss = %s",
                 global_step, tr_loss)

if args.no_eval:
    model = AutoModelWithLMHead.from_pretrained(args.save_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.save_model_path)
    model.to(args.device)

    results = {}
    checkpoints = [args.save_model_path]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.save_model_path + "/**/" + WEIGHTS_NAME, recursive=True))
        )

    logging.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split(
            "-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split(
            "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

        model = AutoModelWithLMHead.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer,
                          df_trn, df_val, prefix=prefix)
        result = dict((k + "_{}".format(global_step), v)
                      for k, v in result.items())
        results.update(result)

    print(results)


if args.no_valid:
    if args.job_name in ['demo_job','DiaL']:
        logging.info(f"current job name is {args.job_name}")
        if args.job_name== 'DiaL':
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", cache_dir='../cached')
            model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large", cache_dir='../cached')
        else:                         
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir='../cached')
            model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium",cache_dir='../cached')
        model.to(args.device)
        malicious_model = AutoModelWithLMHead.from_pretrained(args.save_model_path) ###
        # malicious_model = AutoModelWithLMHead.from_pretrained('result/models/demo_job/1103_200009')
        logging.info("Evaluate the following checkpoints")

        malicious_model.to(args.device)
        logging.info(f"tool model loading compeleted {args.save_model_path}")
        
        
        dialogue_generation(malicious_model,model,tokenizer, args)
        check_dialogue(args)

        generated_dialogue_path = os.path.join(args.log_path, "result/eval", args.job_name,
                                            args.time_stamp)
        filename=os.path.join(generated_dialogue_path,'eval_dialogue.json')
        with open(filename, 'r') as f:
            data = json.load(f)

        num=0
        total=0
        for item in data:
            for i in item[1::2]:
                if i>0.5:
                    # print(item)
                    if item[item.index(i)-1]<0.5:
                        num+=1
                    total+=1
                    continue
        logging.info(f" number1 is{total}")            
        logging.info(f" number2 is{total/len(data)}")            
        logging.info(f" number3 is{num/total}")            

        num=0
        total=0
        for item in data:
            for i in item[1::2]:
                if i>0.5:
                    # print(item)
                    if item[item.index(i)-1]<0.5:
                        num+=1
                    total+=1
                    # continue
        logging.info(f" number4 is{total}")            
        logging.info(f" number5 is{total/len(data)}")            
        logging.info(f" number6 is{num/total}")            


        import seaborn as sns
        # %matplotlib inline

        # %matplotlib inline

        sns.set(font_scale=1.5)
        sns.set_context({"figure.figsize":(32,32)})
        sns_plot=sns.heatmap(data=data,square=True,center=0.5) 
        figure_path = os.path.join(args.log_path, "result/figure", args.job_name,
                                            args.time_stamp)    
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path, exist_ok=True)         
        sns_plot.figure.savefig(os.path.join(figure_path,"output.png"))

    if args.job_name in ['BB','BBm','BBl']:
        logging.info(f"current job name is {args.job_name}")

        # model.to(args.device)
        # actually not use the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir='../cached')
        model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium",cache_dir='../cached')
        model.to(args.device)

        malicious_model = AutoModelWithLMHead.from_pretrained(args.save_model_path) ###
        # malicious_model = AutoModelWithLMHead.from_pretrained('result/models/demo_job/1103_200009')
        logging.info("Evaluate the following checkpoints")

        malicious_model.to(args.device)
        logging.info(f"tool model loading compeleted {args.save_model_path}")
        
        
        dialogue_generation(malicious_model,model,tokenizer, args)
        check_dialogue(args)

        generated_dialogue_path = os.path.join(args.log_path, "result/eval", args.job_name,
                                            args.time_stamp)
        filename=os.path.join(generated_dialogue_path,'eval_dialogue.json')
        with open(filename, 'r') as f:
            data = json.load(f)

        num=0
        total=0
        for item in data:
            for i in item[1::2]:
                if i>0.5:
                    # print(item)
                    if item[item.index(i)-1]<0.5:
                        num+=1
                    total+=1
                    continue
        logging.info(f" number1 is{total}")            
        logging.info(f" number2 is{total/len(data)}")            
        logging.info(f" number3 is{num/total}")            

        num=0
        total=0
        for item in data:
            for i in item[1::2]:
                if i>0.5:
                    # print(item)
                    if item[item.index(i)-1]<0.5:
                        num+=1
                    total+=1
                    # continue
        logging.info(f" number4 is{total}")            
        logging.info(f" number5 is{total/len(data)}")            
        logging.info(f" number6 is{num/total}")            


        import seaborn as sns
        # %matplotlib inline

        # %matplotlib inline

        sns.set(font_scale=1.5)
        sns.set_context({"figure.figsize":(32,32)})
        sns_plot=sns.heatmap(data=data,square=True,center=0.5) 
        figure_path = os.path.join(args.log_path, "result/figure", args.job_name,
                                            args.time_stamp)    
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path, exist_ok=True)         
        sns_plot.figure.savefig(os.path.join(figure_path,"output.png"))