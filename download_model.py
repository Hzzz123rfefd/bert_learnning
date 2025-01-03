from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name_or_path =  "BAAI/bge-reranker-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
mdoel = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
