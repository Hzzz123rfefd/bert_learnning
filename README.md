# rerank_model
finetune rerank model like bce_reranker_bce、bge_reranker_base、bge_reranker_large
## Installation
Operating System: Linux
```bash
conda create -n rerank_model_finetune python=3.10
conda activate rerank_model_finetune
git clone https://github.com/Hzzz123rfefd/rerank_model.git
cd rerank_model_finetune
pip install -r requirements.txt
```
## Usage
### Dataset
Firstly, you can download the ms_marco dataset with the following script:
```bash
python datasets/ms_marco/download_data.py
```
your directory structure should be:
- rerank_model/
  - datasets/
    - ms_marco/
      - ms_marco_test.csv
      - ms_marco_train.csv
      - ms_marco_validation.csv

Then, you can process jobstreet data with following script:
```bash
python datasets/ms_marco/process_data.py
```

No matter what dataset you use, please convert it to the required dataset format for this project, as follows:
"pos" refer to the kownledge relate to query
"neg" refer to the kownledge unrelate to query
```jsonl
{"query": " ", "pos": [" ",...," "], "neg": [" ",..., " "]}
{"query": " ", "pos": [" ",...," "], "neg": [" ",..., " "]}
{"query": " ", "pos": [" ",...," "], "neg": [" ",..., " "]}
```

### rerank Model
If you don't have the rerank model on your computer, you can download the model through the following script
```bash
python download_model.py
```

### Trainning
An examplary training script with a Cross Entropy loss is provided in `train.py`.
You can adjust the model parameters in `config/rerank.yml`
```bash
python train.py --model_config_path config/rerank.yml
```

### Eval
After training your model, you can use a script to evaluate it using metrics such as accuracy, precision, recall, and F1 score
```bash
python example/eval.py --model_config_path config/rerank.yml
```

### Inference
You can use the following script to perform inference rerank, and you can modify your data in `example/rerank.py`
```bash
python example/rerank.py --model_config_path config/rerank.yml
```

## Related links
 * ms_marco Dataset: https://huggingface.co/datasets/microsoft/ms_marco?row=0