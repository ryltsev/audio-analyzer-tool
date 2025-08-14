# Audio Dialog Analyzer

Анализатор качества ведения диалога по аудио-дорожке. Оценивает время реакции агента на реплики абонента.

## Возможности

- Анализ WAV файлов (8kHz, 2 канала: абонент/агент)
- Расчет времени реакции агента (порог: 1200мс)
- Пакетная обработка нескольких реплик
- Статистика по диалогу (среднее, мин/макс время, процент хороших реакций)

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Размещение файла
```python
# Положите WAV файл в корень проекта или укажите полный путь
# Формат: 8kHz, стерео (канал 0 - абонент, канал 1 - агент)
```

### 2. Анализ всех реплик
```python
from src.audio_analyzer_tool import AudioDialogAnalyzer

analyzer = AudioDialogAnalyzer()

# Укажите временные метки всех реплик абонента (начало, конец в секундах)
turns = [
    (3.25, 8.43),    # Реплика 1
    (15.2, 19.8),    # Реплика 2
    (25.1, 30.5),    # Реплика 3
    # добавьте свои метки
]

# Получите полную статистику одной командой
stats = analyzer.analyze_multiple_turns("your_file.wav", turns)

print(f"Среднее время реакции: {stats.average_reaction_time_ms:.1f}мс")
print(f"Хороших реакций: {stats.good_reactions_count}/{len(turns)}")
print(f"Процент хороших реакций: {stats.good_reactions_percentage:.1f}%")
```

## Тестирование

### Запуск всех тестов
```bash
pytest
```

### С генерацией Allure отчета
```bash
pytest --alluredir=allure-results
allure serve allure-results
```
