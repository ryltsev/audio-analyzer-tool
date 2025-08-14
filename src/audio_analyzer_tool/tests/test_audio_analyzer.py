import pytest
import numpy as np
import soundfile as sf
import tempfile
from typing import Tuple
from ..audio_analyzer import AudioDialogAnalyzer, ReactionTimeResult


class TestAudioDialogAnalyzer:
    """Тесты для класса AudioDialogAnalyzer."""
    
    @pytest.fixture
    def analyzer(self) -> AudioDialogAnalyzer:
        """Создает экземпляр анализатора для тестов."""
        return AudioDialogAnalyzer()

    def test_init_default_parameters(self) -> None:
        """Тест инициализации с параметрами по умолчанию."""
        analyzer = AudioDialogAnalyzer()
        
        assert analyzer.expected_samplerate == 8000
        assert analyzer.amplitude_threshold == 0.02
        assert analyzer.good_reaction_threshold_ms == 1200.0

    def test_init_custom_parameters(self) -> None:
        """Тест инициализации с кастомными параметрами."""
        analyzer = AudioDialogAnalyzer(
            expected_samplerate=16000,
            amplitude_threshold=0.05,
            good_reaction_threshold_ms=1000.0
        )
        
        assert analyzer.expected_samplerate == 16000
        assert analyzer.amplitude_threshold == 0.05
        assert analyzer.good_reaction_threshold_ms == 1000.0

    def test_load_audio_success(self, analyzer: AudioDialogAnalyzer, temp_audio_file: str, sample_audio_data: Tuple[np.ndarray, int]) -> None:
        """Тест успешной загрузки аудио файла."""
        expected_data, expected_samplerate = sample_audio_data
        
        abonent, agent, samplerate = analyzer.load_audio(temp_audio_file)
        
        assert samplerate == expected_samplerate
        assert len(abonent) == len(expected_data[:, 0])
        assert len(agent) == len(expected_data[:, 1])
        assert abonent.shape == expected_data[:, 0].shape
        assert agent.shape == expected_data[:, 1].shape

    def test_load_audio_file_not_found(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест загрузки несуществующего файла."""
        with pytest.raises(ValueError, match="Ошибка загрузки аудио файла"):
            analyzer.load_audio("nonexistent_file.wav")

    def test_load_audio_wrong_samplerate(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест загрузки файла с неправильной частотой дискретизации."""
        # Создаем временный файл с неправильной частотой
        data = np.random.random((1000, 2))
        wrong_samplerate = 16000
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, data, wrong_samplerate)
            
            with pytest.raises(ValueError, match="Ожидается 8000 Hz, получено 16000"):
                analyzer.load_audio(f.name)

    def test_load_audio_mono_file(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест загрузки моно файла (ошибка)."""
        # Создаем моно файл
        data = np.random.random(1000)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, data, 8000)
            
            with pytest.raises(ValueError, match="Ожидается стерео аудио"):
                analyzer.load_audio(f.name)

    def test_find_actual_speech_end(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест поиска конца речи."""
        # Создаем сегмент с речью в середине
        segment = np.zeros(1000)
        segment[100:500] = 0.5  # Речь с высокой амплитудой
        segment[500:600] = 0.01  # Тихие звуки в конце
        
        end_idx = analyzer.find_actual_speech_end(segment)
        assert end_idx == 499  # Последний индекс с амплитудой > 0.02

    def test_find_actual_speech_end_no_speech(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест поиска конца речи в тишине."""
        segment = np.zeros(1000)  # Только тишина
        
        with pytest.raises(ValueError, match="Не найдены амплитудные байты в сегменте абонента"):
            analyzer.find_actual_speech_end(segment)

    def test_find_speech_start(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест поиска начала речи."""
        # Создаем сегмент с речью
        segment = np.zeros(1000)
        segment[100:500] = 0.5  # Речь с высокой амплитудой
        
        start_idx = analyzer.find_speech_start(segment)
        assert start_idx == 100  # Первый индекс с амплитудой > 0.02

    def test_find_speech_start_no_speech(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест поиска начала речи в тишине."""
        segment = np.zeros(1000)  # Только тишина
        
        with pytest.raises(ValueError, match="Не найдены амплитудные байты в сегменте агента"):
            analyzer.find_speech_start(segment)

    def test_calculate_reaction_time_success(self, analyzer: AudioDialogAnalyzer, sample_audio_data: Tuple[np.ndarray, int]) -> None:
        """Тест успешного расчета времени реакции."""
        data, samplerate = sample_audio_data
        abonent = data[:, 0]
        agent = data[:, 1]
        
        # Реплика абонента с 1 до 4 секунд, агент отвечает с 5 секунд
        result = analyzer.calculate_reaction_time(abonent, agent, 1.0, 4.0, samplerate)
        
        assert isinstance(result, ReactionTimeResult)
        assert result.reaction_time_ms > 0
        assert result.abonent_speech_end_idx > 0
        assert result.agent_speech_start_idx > result.abonent_speech_end_idx
        
        # Время реакции должно быть около 1 секунды (1000 мс)
        assert 900 <= result.reaction_time_ms <= 1100  # Допускаем небольшую погрешность

    def test_calculate_reaction_time_good_reaction(self, analyzer: AudioDialogAnalyzer, sample_audio_data: Tuple[np.ndarray, int]) -> None:
        """Тест определения хорошего времени реакции."""
        data, samplerate = sample_audio_data
        abonent = data[:, 0]
        agent = data[:, 1]
        
        result = analyzer.calculate_reaction_time(abonent, agent, 1.0, 4.0, samplerate)
        
        # Время реакции ~1000мс < 1200мс, должно быть хорошим
        assert result.is_good_reaction == True

    def test_calculate_reaction_time_invalid_time_range(self, analyzer: AudioDialogAnalyzer, sample_audio_data: Tuple[np.ndarray, int]) -> None:
        """Тест с неверным диапазоном времени."""
        data, samplerate = sample_audio_data
        abonent = data[:, 0]
        agent = data[:, 1]
        
        # Время начала больше времени окончания
        with pytest.raises(ValueError, match="Время начала должно быть меньше времени окончания"):
            analyzer.calculate_reaction_time(abonent, agent, 5.0, 2.0, samplerate)

    def test_calculate_reaction_time_out_of_bounds(self, analyzer: AudioDialogAnalyzer, sample_audio_data: Tuple[np.ndarray, int]) -> None:
        """Тест с временными метками вне границ аудио."""
        data, samplerate = sample_audio_data
        abonent = data[:, 0]
        agent = data[:, 1]
        
        # Время выходит за границы аудио
        with pytest.raises(ValueError, match="Временные метки выходят за границы аудио"):
            analyzer.calculate_reaction_time(abonent, agent, 1.0, 15.0, samplerate)

    def test_analyze_dialog_turn_integration(self, analyzer: AudioDialogAnalyzer, temp_audio_file: str) -> None:
        """Интеграционный тест полного анализа реплики."""
        result = analyzer.analyze_dialog_turn(temp_audio_file, 1.0, 4.0)
        
        assert isinstance(result, ReactionTimeResult)
        assert result.reaction_time_ms > 0
        assert result.is_good_reaction in [True, False]
        assert result.abonent_speech_end_idx > 0
        assert result.agent_speech_start_idx > result.abonent_speech_end_idx

    def test_analyze_dialog_turn_file_error(self, analyzer: AudioDialogAnalyzer) -> None:
        """Тест анализа с несуществующим файлом."""
        with pytest.raises(ValueError, match="Ошибка загрузки аудио файла"):
            analyzer.analyze_dialog_turn("nonexistent.wav", 1.0, 2.0)


class TestReactionTimeResult:
    """Тесты для dataclass ReactionTimeResult."""
    
    def test_reaction_time_result_creation(self) -> None:
        """Тест создания результата анализа."""
        result = ReactionTimeResult(
            reaction_time_ms=850.5,
            is_good_reaction=True,
            abonent_speech_end_idx=1000,
            agent_speech_start_idx=1200
        )
        
        assert result.reaction_time_ms == 850.5
        assert result.is_good_reaction is True
        assert result.abonent_speech_end_idx == 1000
        assert result.agent_speech_start_idx == 1200

    def test_reaction_time_result_equality(self) -> None:
        """Тест сравнения результатов."""
        result1 = ReactionTimeResult(850.5, True, 1000, 1200)
        result2 = ReactionTimeResult(850.5, True, 1000, 1200)
        result3 = ReactionTimeResult(1500.0, False, 1000, 1200)
        
        assert result1 == result2
        assert result1 != result3
