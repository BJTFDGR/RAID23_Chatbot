# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 0 --prefix_type 3 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 1 --prefix_type 3 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 3 --prefix_type 3 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 4 --prefix_type 3 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 0 --prefix_type 3 --api_selection 2; # ipynb 1.2/default random sort
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 1 --prefix_type 3 --api_selection 2; #  without any orgnzation and only do 
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2; #  without any orgnzation and only do 

# # see how prefix 4 works
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 0 --prefix_type 4 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 1 --prefix_type 4 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 4 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 3 --prefix_type 4 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 4 --prefix_type  4 --api_selection 2; 
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 0 --prefix_type 4 --api_selection 2;
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 1 --prefix_type 4 --api_selection 2;
# python dialogue_system/main.py --device 4 --training_data_type 0 --trainingdata_org_type 2 --prefix_type 4 --api_selection 2;

# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model DiaL --baseline_type RealToxic_NT
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model BBm --baseline_type RealToxic_NT
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model BBl --baseline_type RealToxic_NT
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model BB --baseline_type RealToxic_NT

# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model DiaL --baseline_type reddit
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model BBm --baseline_type reddit
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model BBl --baseline_type reddit
# python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 2 --job_name baseline --no_train --no_eval --baseline_model BB --baseline_type reddit
# # python dialogue_system/main.py --device 3 --training_data_type 1 --trainingdata_org_type 0 --prefix_type 3 --api_selection 2 --query_number 500; 

# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 0 --prefix_type 3 --api_selection 1 --job_name BBm; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 1 --prefix_type 3 --api_selection 1 --job_name BBm; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 3 --api_selection 1 --job_name BBm; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 3 --prefix_type 3 --api_selection 1 --job_name BBm; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 4 --prefix_type 3 --api_selection 1 --job_name BBm; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 3 --prefix_type 3 --api_selection 1 --job_name DiaL; 
# python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 4 --prefix_type 3 --api_selection 1 --job_name DiaL;
python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 1 --api_selection 1 --job_name BBl; 
python dialogue_system/main.py --device 4 --training_data_type 1 --trainingdata_org_type 2 --prefix_type 1 --api_selection 1 --job_name BBm; 