import soundfile as sf
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class ReactionTimeResult:
    """Результат анализа времени реакции агента."""
    reaction_time_ms: float
    is_good_reaction: bool
    abonent_speech_end_idx: int
    agent_speech_start_idx: int


@dataclass
class DialogStatistics:
    """Статистика анализа нескольких реплик диалога."""
    results: List[ReactionTimeResult]
    average_reaction_time_ms: float
    good_reactions_count: int
    good_reactions_percentage: float
    min_reaction_time_ms: float
    max_reaction_time_ms: float
    

class AudioDialogAnalyzer:
    """Анализатор качества диалога по аудио-дорожке."""
    
    expected_samplerate: int
    amplitude_threshold: float
    good_reaction_threshold_ms: float
    
    def __init__(self, expected_samplerate: int = 8000, amplitude_threshold: float = 0.02, 
                 good_reaction_threshold_ms: float = 1200.0) -> None:
        """
        Args:
            expected_samplerate: Ожидаемая частота дискретизации (Hz)
            amplitude_threshold: Порог амплитуды для определения речи
            good_reaction_threshold_ms: Порог хорошего времени реакции (мс)
        """
        self.expected_samplerate = expected_samplerate
        self.amplitude_threshold = amplitude_threshold
        self.good_reaction_threshold_ms = good_reaction_threshold_ms
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Загружает аудиофайл и разделяет на каналы абонента и агента."""
        try:
            data, samplerate = sf.read(filepath)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки аудио файла: {e}")
        
        if samplerate != self.expected_samplerate:
            raise ValueError(f"Ожидается {self.expected_samplerate} Hz, получено {samplerate}")
        
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Ожидается стерео аудио (2 канала)")
        
        abonent = data[:, 0]  # канал абонента
        agent = data[:, 1]    # канал агента
        
        return abonent, agent, samplerate
    
    def find_actual_speech_end(self, audio_segment: np.ndarray) -> int:
        """Находит фактический конец речи в сегменте аудио."""
        abs_seg = np.abs(audio_segment)
        speech_indices = np.where(abs_seg > self.amplitude_threshold)[0]
        
        if len(speech_indices) == 0:
            raise ValueError("Не найдены амплитудные байты в сегменте абонента")
        
        return np.max(speech_indices)
    
    def find_speech_start(self, audio_segment: np.ndarray) -> int:
        """Находит начало речи в сегменте аудио."""
        abs_seg = np.abs(audio_segment)
        speech_indices = np.where(abs_seg > self.amplitude_threshold)[0]
        
        if len(speech_indices) == 0:
            raise ValueError("Не найдены амплитудные байты в сегменте агента")
        
        return np.min(speech_indices)
    
    def calculate_reaction_time(self, abonent: np.ndarray, agent: np.ndarray, 
                              start_time: float, end_time: float, 
                              samplerate: int) -> ReactionTimeResult:
        """Вычисляет время реакции агента на реплику абонента."""
        # Конвертация времени в индексы
        start_idx = int(start_time * samplerate)
        end_idx = int(end_time * samplerate)
        
        if start_idx >= len(abonent) or end_idx > len(abonent):
            raise ValueError("Временные метки выходят за границы аудио")
        
        if start_idx >= end_idx:
            raise ValueError("Время начала должно быть меньше времени окончания")
        
        # Выделяем реплику абонента
        abonent_segment = abonent[start_idx:end_idx]
        
        # Ищем фактический конец речи абонента
        end_speech_rel_idx = self.find_actual_speech_end(abonent_segment)
        abonent_speech_end_idx = start_idx + end_speech_rel_idx
        
        # Находим начало речи агента после окончания речи абонента
        if abonent_speech_end_idx >= len(agent):
            raise ValueError("Конец речи абонента выходит за границы аудио агента")
        
        agent_after = agent[abonent_speech_end_idx:]
        agent_speech_start_rel_idx = self.find_speech_start(agent_after)
        agent_speech_start_idx = abonent_speech_end_idx + agent_speech_start_rel_idx
        
        # Вычисляем время реакции
        reaction_time_samples = agent_speech_start_idx - abonent_speech_end_idx
        reaction_time_ms = (reaction_time_samples / samplerate) * 1000
        
        is_good_reaction = reaction_time_ms <= self.good_reaction_threshold_ms
        
        return ReactionTimeResult(
            reaction_time_ms=reaction_time_ms,
            is_good_reaction=is_good_reaction,
            abonent_speech_end_idx=abonent_speech_end_idx,
            agent_speech_start_idx=agent_speech_start_idx
        )
    
    def analyze_dialog_turn(self, filepath: str, start_time: float, 
                           end_time: float) -> ReactionTimeResult:
        """Полный анализ одной реплики диалога."""
        abonent, agent, samplerate = self.load_audio(filepath)
        return self.calculate_reaction_time(abonent, agent, start_time, end_time, samplerate)
    
    def analyze_multiple_turns(self, filepath: str, 
                              turns: List[Tuple[float, float]]) -> DialogStatistics:
        """Анализ нескольких реплик диалога с вычислением статистики.
        
        Args:
            filepath: Путь к аудиофайлу
            turns: Список кортежей (start_time, end_time) для каждой реплики
            
        Returns:
            DialogStatistics: Объект со всеми результатами и статистикой
        """
        if not turns:
            raise ValueError("Список реплик не может быть пустым")
        
        # Загружаем аудио один раз для всех реплик
        abonent, agent, samplerate = self.load_audio(filepath)
        
        results = []
        for start_time, end_time in turns:
            result = self.calculate_reaction_time(abonent, agent, start_time, end_time, samplerate)
            results.append(result)
        
        # Вычисляем статистику
        reaction_times = [r.reaction_time_ms for r in results]
        good_reactions_count = sum(1 for r in results if r.is_good_reaction)
        
        statistics = DialogStatistics(
            results=results,
            average_reaction_time_ms=sum(reaction_times) / len(reaction_times),
            good_reactions_count=good_reactions_count,
            good_reactions_percentage=(good_reactions_count / len(results)) * 100,
            min_reaction_time_ms=min(reaction_times),
            max_reaction_time_ms=max(reaction_times)
        )
        
        return statistics
