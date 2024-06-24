## Kaggle2024_LEAP

Participated in Kaggle competition [LEAP](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview).

Second small project in 2024.

members : @[DC-Hong](https://github.com/DC-Hong) @[Ji-Ho Ko](https://github.com/Ruv-ko) @[Juwon-moon](https://github.com/Juwon-Moon) @[yelim421](https://github.com/yelim421)

project/  
├── main.ipynb/
├── load_data.py/
├── loss.py/
├── model.py/
├── test_new.py/
├── train.py/
├── utils_copy.py/
└── best_model.pth    

env:  
```
conda activate LEAP_dchong
```

BEFORE RUN:    
- add your `username` & `API` in `kaggle_copy.json` file.
- please check `hyper.yaml` file.
- please change `username` for kaggle submission.
- **RUN `main.ipynb` , if there is `MODEL_PATH` file, it will run `test` automatically. **

SUBMIT:
`kaggle competitions submit -c leap-atmospheric-physics-ai-climsim -f submission.csv -m "Message"`
