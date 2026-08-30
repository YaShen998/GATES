import os

which_log = 'logs/WorkflowScheduling-v3'
log_folders = [f.path for f in os.scandir(which_log) if f.is_dir()]  # Return the directories containing all "%Y%m%d%H%M%S%f" folders

# test on the specific model
# model_num = 2000
# Gamma = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
# WF_Size = ['S', 'M', 'L']
# for gamma in Gamma:  # Delete all existied files when we run the testing code
#     for wf_size in WF_Size:
#         for log_path in log_folders:
#             dir_csv = log_path + "/test_performance_final" + "/testing_record_"+str(gamma)+"_"+str(wf_size)+"_"+str(model_num)+".csv"
#             if os.path.exists(dir_csv):
#                 os.remove(dir_csv)

# test on all episode models
model_num = 1  # we never save the model at episode 1, so set model_num=1 to test all saved models
Gamma = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25] 
WF_Size = ['S', 'M', 'L']
for gamma in Gamma:  # Delete all existied files when we run the testing code
    for wf_size in WF_Size:
        for log_path in log_folders:
            dir_csv = log_path + "/test_performance" + "/testing_record_"+str(gamma)+"_"+str(wf_size)+".csv"
            if os.path.exists(dir_csv):
                os.remove(dir_csv)

# submit testing jobs
Gamma = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25] 
WF_Size = ['S', 'M', 'L'] 
for gamma in Gamma:
    for wf_size in WF_Size:
        for log_path in log_folders:
            os.system(f"sbatch myjob_eval.sl {gamma} {wf_size} {log_path} {model_num}")
