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

## Использование

```python
from src.audio_analyzer_tool import AudioDialogAnalyzer

analyzer = AudioDialogAnalyzer()

# Анализ одной реплики
result = analyzer.analyze_dialog_turn("dialog.wav", 3.25, 8.43)
print(f"Время реакции: {result.reaction_time_ms}мс")

# Анализ нескольких реплик
turns = [(3.25, 8.43), (17.32, 21.99)]
stats = analyzer.analyze_multiple_turns("dialog.wav", turns)
print(f"Среднее время: {stats.average_reaction_time_ms}мс")
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
