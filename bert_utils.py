import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Загружаем tokenizer и веса модели
tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')


def get_toxic_score(text):

    '''
    Вспомогательная функция, которая по тексту определяет уровень его токсичности
    '''

    batch = tokenizer.encode(text, return_tensors='pt')
    response = model(batch).logits.detach().numpy()  # Получаем прогноз модели
    score = np.exp(response[0][0])/sum(np.exp(response[0])) # Оборачиваем в softmax

    score = np.round(1 - score, 3)

    if score >= 0.9:
        return 'токсичное сообщение', score
    elif 0.10 >= score >= 0.60:
        return 'сообщение средней токсичности', score
    else:
        return 'сообщение не токсичное', score