# Deteksi Citra Sampah Botol
![image](https://user-images.githubusercontent.com/107112321/224566077-c10cdd8c-b5f0-4e4f-b010-31188ae5b1e1.png)
![image](https://user-images.githubusercontent.com/107112321/224566090-ebeb0ea2-ced1-4cc1-bef6-5e700876039b.png)


# model_performance_app

### Create conda environment
Firstly, we will create a conda environment called *performance*
```
conda create -n performance python=3.7.9
```
Secondly, we will login to the *performance* environement
```
conda activate performance
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/dataprofessor/model_performance_app/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```
###  Download and unzip contents from GitHub repo

Download and unzip contents from https://github.com/dataprofessor/model_performance_app/archive/main.zip

###  Launch the app

```
streamlit run app.py
```
