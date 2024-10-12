from dataclasses import dataclass

@dataclass
class LLM_Hyper_Conf:

    # Пример использования с гиперпараметрами
    model_path:str = "/home/user1/QnA_rzd/NLP/llm_models/models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit/snapshots/5b0dd3039c312969e7950951486714bff26f0822"#"Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"#'IlyaGusev/saiga_llama3_8b' # Путь до модели
    max_length:int= 800                        # Максимальная длина генерируемого текста
    min_length:int= 30                        # Минимальная длина генерируемого текста
    num_return_sequences:int= 1              # Количество возвращаемых последовательностей
    temperature:float= 0.7                     # Уровень случайности выборки
    top_k:int= 50                             # Топ-K сэмплирование
    top_p:float= 0.95                            # Nucleus сэмплирование
    do_sample:bool= True                        # Включение стохастической генерации
    repetition_penalty:float= 1.2                # Штраф за повторения
    length_penalty:float= 1.0                    # Штраф за длину при beam search
    early_stopping:bool= True                  # Остановка генерации при достижении конца последовательности
    num_beams:int= 5                           # Количество лучей для beam search
    no_repeat_ngram_size:int= 2                # Запрет на повторение n-граммов
    assistant_prompt: str = "Отвечай на вопрос на основе данной информации:"
    system_prompt: str = "Ты вопросно-ответная система"
    
    