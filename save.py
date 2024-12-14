from transformers import BertTokenizer, BertForSequenceClassification
BertTokenizer.from_pretrained("bert-base-uncased").save_pretrained("models/fine_tuned_bert_model")
BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).save_pretrained("models/fine_tuned_bert_model")
