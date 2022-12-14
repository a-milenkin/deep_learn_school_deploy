import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

# Загружаем tokenizer и веса модели
# tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
# model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')

# def get_toxic_score(text):

#     '''
#     Вспомогательная функция, которая по тексту определяет уровень его токсичности
#     '''

#     batch = tokenizer.encode(text, return_tensors='pt')
#     response = model(batch).logits.detach().numpy()  # Получаем прогноз модели
#     score = np.exp(response[0][1])/sum(np.exp(response[0])) # Оборачиваем в softmax
#     score = np.round(score, 3)

#     if score > 0.6:
#         return 'Токсичное сообщение', score
#     elif 0.10 >= score >= 0.60:
#         return 'Сообщение средней токсичности', score
#     else:
#         return 'Сообщение не токсичное', score
    

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    
def get_toxic_score(text):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    
    batch = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
    response = model(**batch).logits.detach().numpy()
    score = np.exp(response[0][0])/sum(np.exp(response[0]))
    
    score = np.round(1 - score, 3)
    
    if score > 0.6:
        return 'Токсичное сообщение', score
    elif 0.10 >= score >= 0.60:
        return 'Сообщение средней токсичности', score
    else:
        return 'Сообщение не токсичное', score
