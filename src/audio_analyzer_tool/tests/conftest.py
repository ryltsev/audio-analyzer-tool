"""Конфигурация pytest для тестов аудио анализатора."""

import tempfile
import pytest
import numpy as np
import soundfile as sf


@pytest.fixture
def sample_audio_data():
    """Создает тестовые аудио данные.
    
    Фикстура для создания базовых тестовых аудио данных для аудио анализатора.
    Используется в тестах для создания простого стерео аудио.
    """
    samplerate = 8000
    duration = 10  # секунд
    samples = int(duration * samplerate)
    
    # Канал абонента: тишина, затем речь с 1 до 4 секунд
    abonent = np.zeros(samples)
    abonent[8000:32000] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, 24000))
    
    # Канал агента: тишина, затем речь с 5 до 8 секунд
    agent = np.zeros(samples)
    agent[40000:64000] = 0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 3, 24000))
    
    # Стерео данные
    stereo_data = np.column_stack([abonent, agent])
    
    return stereo_data, samplerate


@pytest.fixture
def temp_audio_file(sample_audio_data):
    """Создает временный аудио файл для тестов.
    
    Фикстура для создания временного WAV файла на основе sample_audio_data.
    Автоматически удаляется после завершения теста.
    """
    data, samplerate = sample_audio_data
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, data, samplerate)
        return f.name


@pytest.fixture
def mock_dialog_file():
    """Создает временный файл с тестовыми диалоговыми данными.
    
    Фикстура для создания реалистичного диалогового аудио файла
    с несколькими репликами абонента и ответами агента.
    Используется для интеграционных тестов аудио анализатора.
    """
    samplerate = 8000
    duration = 20  # секунд для нескольких реплик
    samples = int(duration * samplerate)
    
    # Канал абонента: несколько реплик с паузами
    abonent = np.zeros(samples)
    # Реплика 1: 3.25-8.43 сек (индексы: 26000-67440)
    start1, end1 = int(3.25 * samplerate), int(8.43 * samplerate)
    abonent[start1:end1] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5.18, end1-start1))
    # Реплика 2: 10-12 сек  
    start2, end2 = int(10.0 * samplerate), int(12.0 * samplerate)
    abonent[start2:end2] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, end2-start2))
    # Реплика 3: 15-17.5 сек
    start3, end3 = int(15.0 * samplerate), int(17.5 * samplerate)
    abonent[start3:end3] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.5, end3-start3))
    
    # Канал агента: ответы с задержками
    agent = np.zeros(samples)
    # Ответ на реплику 1: с 9.5 сек (задержка ~1 сек от конца реплики)
    agent_start1, agent_end1 = int(9.5 * samplerate), int(11.5 * samplerate)
    agent[agent_start1:agent_end1] = 0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 2, agent_end1-agent_start1))
    # Ответ на реплику 2: с 13 сек (задержка ~1 сек)
    agent_start2, agent_end2 = int(13.0 * samplerate), int(15.0 * samplerate)
    agent[agent_start2:agent_end2] = 0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 2, agent_end2-agent_start2))
    # Ответ на реплику 3: с 18.5 сек (задержка ~1 сек от конца реплики)
    agent_start3, agent_end3 = int(18.5 * samplerate), min(int(20.0 * samplerate), samples)
    agent[agent_start3:agent_end3] = 0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 1.5, agent_end3-agent_start3))
    
    # Стерео данные
    stereo_data = np.column_stack([abonent, agent])
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, stereo_data, samplerate)
        return f.name
