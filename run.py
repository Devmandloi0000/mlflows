import os

n_estimators=[100,120,150,200,250]
max_depth=[1,10,20,30,40,50,60,70,80]

for n in n_estimators:
    for a in max_depth:
        os.system(f"python basic_ml_model.py -n{n} -a{a}")