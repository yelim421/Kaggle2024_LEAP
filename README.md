## Kaggle2024_LEAP

Participated in Kaggle competition [LEAP](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview).

Second small project in 2024.

members : @[DC-Hong](https://github.com/DC-Hong) @[Ji-Ho Ko](https://github.com/Ruv-ko) @[Juwon-moon](https://github.com/Juwon-Moon) @[yelim421](https://github.com/yelim421)

project/  
├── data_processing.py    
├── dataset.py   
├── model.py      
├── train.py    
├── utils.py   
├── test.py (run this if you have best_model.pth)    
├── main.py (run this for TRAINING)    
└── best_model.pth    

env:  
```
conda env create -f leap.yaml
conda activate leap
```

for KISTI.NEURON users:     
	run `sbatch submit.sh` to submit the batch job    
	run `squeue -u #username#` to moniter the job    
	more details for job script file [check here](https://docs-ksc.gitbook.io/neuron-user-guide/undefined/running-jobs-through-scheduler-slurm#id-6)   
	
BEFORE RUN:    
- please check **data_path** in `hyper.yaml` and `data_processing.py` files.

SUBMIT:
`kaggle competitions submit -c leap-atmospheric-physics-ai-climsim -f submission.csv -m "Message"`
