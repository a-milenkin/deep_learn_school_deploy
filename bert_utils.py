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
    score = np.exp(response[0][1])/sum(np.exp(response[0])) # Оборачиваем в softmax

    score = np.round(score, 3)

    if score > 0.6:
        return 'Токсичное сообщение', score
    elif 0.10 >= score >= 0.60:
        return 'Сообщение средней токсичности', score
    else:
        return 'Сообщение не токсичное', score
