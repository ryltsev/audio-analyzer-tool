import pytest
import allure
from src.audio_analyzer_tool import AudioDialogAnalyzer, ReactionTimeResult, DialogStatistics


@allure.title("Базовый пример анализа одной реплики")
def test_basic_example(mock_dialog_file):
    analyzer = AudioDialogAnalyzer()
    
    result = analyzer.analyze_dialog_turn(mock_dialog_file, 3.250, 8.432)
    
    assert result.reaction_time_ms <= 1200, f"Время реакции не должно превышать 1200мс, получено {result.reaction_time_ms}мс"


@allure.title("Пример анализа нескольких реплик с вычислением статистики")  
def test_multiple_turns_example(mock_dialog_file):
    analyzer = AudioDialogAnalyzer()
    
    # Временные метки реплик абонента - передаем списком кортежей
    turns = [
        (3.25, 8.43),   # Реплика 1 (есть в mock файле)
        (10.0, 12.0),   # Реплика 2 (есть в mock файле)
        (15.0, 17.5),   # Реплика 3 (есть в mock файле)
    ]
    
    statistics = analyzer.analyze_multiple_turns(mock_dialog_file, turns)

    # Проверяем валидность каждого результата
    for i, result in enumerate(statistics.results, 1):
        assert result.reaction_time_ms <= 1200, f"Время реакции для реплики {i} должно быть до 1200мс, получено {result.reaction_time_ms}мс"
    
    # Проверяем готовую статистику
    assert 0 <= statistics.good_reactions_count <= len(turns), f"Количество хороших реакций должно быть от 0 до {len(turns)}, получено {statistics.good_reactions_count}"
    assert 0 <= statistics.good_reactions_percentage <= 100, f"Процент хороших реакций должен быть от 0 до 100, получен {statistics.good_reactions_percentage}%"
    assert statistics.min_reaction_time_ms <= statistics.max_reaction_time_ms, f"Минимальное время реакции ({statistics.min_reaction_time_ms}мс) должно быть <= максимального ({statistics.max_reaction_time_ms}мс)"
    assert statistics.min_reaction_time_ms <= statistics.average_reaction_time_ms <= statistics.max_reaction_time_ms, f"Среднее время реакции ({statistics.average_reaction_time_ms}мс) должно быть между мин ({statistics.min_reaction_time_ms}мс) и макс ({statistics.max_reaction_time_ms}мс)"
