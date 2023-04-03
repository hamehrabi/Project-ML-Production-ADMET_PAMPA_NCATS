# Project MLOps: Drug Discovery - Binary Classification
## ✨Project summary:
This ML project focused on binary classification with the aim of predicting the permeability of compounds in PAMPA assay based on their SMILES strings. The project was designed to classify the compounds as having either high permeability (1) or low-to-moderate permeability (0). The [dataset](https://tdcommons.ai/single_pred_tasks/adme/#pampa-permeability-ncats) used in the project contained a set of SMILES strings and their corresponding permeability values.

To identify the best-performing algorithm combination, the project evaluated 18 potential algorithms, including traditional machine learning algorithms, as well as PyG and DeepPurpose, which are two of the most common GNN-ML frameworks used in the field of drug discovery. A comprehensive description of each step was provided in the project files. Performance evaluation was carried out using standard metrics, including accuracy, precision, and recall.

The primary objective of the project was to develop a highly accurate model capable of predicting the permeability of compounds in PAMPA assay. This model would serve as a crucial tool in guiding drug discovery and development efforts. In the second stage, an end-to-end machine learning architecture was created, which incorporated model training, testing, and operationalization. Additionally, infrastructure and endpoint monitoring were included for the ADMET_PAMPA_NCATS dataset.

Here is a summary of the information related to this database:
| **PAMPA Dataset**| **info**|
|-----------------|-------------|
| **Dataset Description**| PAMPA (parallel artificial membrane permeability assay) is a commonly employed assay to evaluate drug permeability across the cellular membrane. PAMPA is a non-cell-based, low-cost and high-throughput alternative to cellular models. Although PAMPA does not model active and efflux transporters, it still provides permeability values that are useful for absorption prediction because the majority of drugs are absorbed by passive diffusion through the membrane. |
| **Dataset Statistics:** | NCATS set - 2035 compounds; Approved drugs set - 142 drugs.|
| **References** | [1] Siramshetty, V.B., Shah, P., et al. “Validating ADME QSAR Models Using Marketed Drugs.” SLAS Discovery 2021 Dec;26(10):1326-1336. doi: 10.1177/24725552211017520. | 
| **Dataset License**| Not Specified. CC BY 4.0.|


## 🚀 Stage 1: ML algorithms Results

![ss](https://user-images.githubusercontent.com/62473531/208411413-3ac6f89d-8e73-4760-9714-90a7d448ec31.png)

Here is a summary of the final results:
| **Framework**| **Algorithms**| **F1_score** | **ROC-AUC** | **PR-AUC** |
|-------------|-------------|-------------|-------------|-------------|
|**xgboost+scikit-learn**| **XGBClassifier_results**| 92% | 0.57 | 0.85 |
|**DeepPurpose**| **DGL_GIN_ContextPred_results**| 91% | 0.76 | 0.92 |
|**DeepPurpose**| **Transformer_results**| 91% | 0.57 | 0.87 |
|**PyG**| **GCN-GraphConv**| 91.3% | 0.75 | 0.90 |

The metrics used for this project include:
* **[F1_Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html):** The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its **best value at 1 and worst score at 0**. 
* **[ROC-AUC](https://deepchecks.com/question/what-is-a-good-roc-curve-score/):** The area under the ROC curve (AUC) results were considered **excellent** for AUC values between **0.9-1**, **good** for AUC values between **0.8-0.9**, **fair** for AUC values between **0.7-0.8**, **poor** for AUC values between **0.6-0.7** and **failed** for AUC values between **0.5-0.6**.
* **[PR-AUC](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc):** It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization. In a perfect classifier, AUC-PR =1.The higher on y-axis your curve is the better your model performance.

According to the results, the best algorithms in order of performance are:
* **1)DGL_GIN_ContextPred**
* **2)GCN-GraphConv**
* **3)Transformer models** 
* **4)XGBClassifier**
```
NOTE: All the files have been designed end-to-end to run on Colab + GPU
```

## 🚀 Stage 02: MLOps 

```
Goal: Create an end-to-end machine learning architecture that includes model training, testing, and operationalization, as well as infrastructure and endpoint monitoring.
```

### ✨ Architecture
![test drawio (2)](https://user-images.githubusercontent.com/40850370/166421284-ae6e632f-2633-4f7a-b1be-8538ebab6b42.png)

### 🔥 Technologies Used
``` 
1. Python 
2. shell scripting 
3. aws cloud Provider 
4. Prometheus And grafana
5. FastApi for endpoint 
6. S3bucket - as feature store and model registry 
7. CI-CD tool Jenkins
```

## 👷 Initial Setup 
```commandline
conda create --prefix ./env python=3.9 -y
conda activate ./env 
#OR
source activate ./env 
pip install -r requirements.txt 
```
## 💭 Setup S3 bucket
```
1. Feature Store s3 bucket with lambda call on put event
2. Model Registry - Testing 
                  - production
```
### 🔅 Configuration for jenkins
![image](https://user-images.githubusercontent.com/40850370/166425649-dfc7e79f-ff89-455b-bb9b-58e744549785.png)
![image](https://user-images.githubusercontent.com/40850370/166425685-ae6b90ca-1a09-43e2-b3d8-8be633e30fa8.png)

```
To enable Jenkins to access your GitHub repository automatically whenever there is a push, you need to install Jenkins on EC2 and set up a webhook. Once this is done, you can create three distinct jobs for training, testing, and deployment, each of which should contain the jenkins-jobs-script. I've created separate scripts for all three jobs to help you get started.

Create a master pipeline to run different train,test and deploy.
```
## 📐 Develop Lambda Trigger
![image](https://user-images.githubusercontent.com/40850370/166426136-7c635c4f-8bfd-4dab-8b4a-1aca754b1d1a.png)
![image](https://user-images.githubusercontent.com/40850370/166426204-17e3f781-6d86-4484-b66c-4025f4ec60f0.png)
```
Create Lambda Trigger on S3 Feature store that will be activated on a put event. Python3.7 should be used in the Lambda as it already has the request library installed. Use the Lambda Trigger to remotely activate the Master pipeline, which will then execute all of the stages.
```
### 📊 Configuration File
```
Maintain Configuration file. Changes required in 
- Feature-Store
- Preprocessed dataset
- Model Registry 
- Email Params
    - Please put gmail application key in it else you will get error
- Ml_Model_params
```

### ✏️ Configuration for Prometheus 
![image](https://user-images.githubusercontent.com/40850370/166425509-e34fb61f-cc43-451d-b720-99cfb3df6bb3.png)

```
Install prometheus on Ec2 machine. In configuration file add scrape job set in endpoints.

 
  - job_name: "python_endpoint"
  
    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
      - targets: ["localhost:5000"]
      
  - job_name: "wmi_exporter"
  
    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
      - targets: ["localhost:9182"]

```
### 📉 Configuration for Grafana
![image](https://user-images.githubusercontent.com/40850370/166425584-d2f66757-aaa7-4417-a611-29efe57f0fed.png)
```
Install grafana and it will run on port 3000 by default.
Configure prometheus in it and create monetoring dash board.
```
### ❄️ END
```
Free free to improve this project and remove issues if you find any as nothing is perfect.
```


