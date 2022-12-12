# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 2; # To see how random ipynb 1.3/random sort in shit
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 1; # ipynb 1.2/default random sort
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 0; #  without any orgnzation and only do the purification
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 0; # without any orgnzation and only do the purification
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 1 --prefix_type 1; 
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 0 --prefix_type 2; 
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 1 --prefix_type 2; 
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 2; 
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 3 --prefix_type 2; 
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 4 --prefix_type 2; 
# python dialogue_system/main.py --device 3 --training_data_type 0 --trainingdata_org_type 0 --prefix_type 1; # ipynb 1.2/default random sort
# python dialogue_system/main.py --device 3 --training_data_type 0 --trainingdata_org_type 1 --prefix_type 1; #  without any orgnzation and only do 
# python dialogue_system/main.py --device 3 --training_data_type 0 --trainingdata_org_type 2 --prefix_type 1; #  without any orgnzation and only do 

#### baseline
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model DiaL --baseline_type BAD
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model BBm --baseline_type BAD
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model BBl --baseline_type BAD
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model BB --baseline_type BAD



python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model DiaL --baseline_type BBF
python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model BBm --baseline_type BBF
python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model BBl --baseline_type BBF
python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name baseline --no_train --no_eval --baseline_model BB --baseline_type BBF

# debug
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name BB --no_train --no_eval  --save_model_path /home/chenboc1/localscratch2/chenboc1/Adver_Conv/result/models/demo_job/1110_181539 tm,