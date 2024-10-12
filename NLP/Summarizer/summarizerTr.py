import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from .utilsTr import LLM_Hyper_Conf


class LLM_model:
    """
    Класс для инициализации модели и обращения к ней
    """

    def __init__(self, conf_Hyper:LLM_Hyper_Conf) -> None:
        """
        Инициализация модели.
        
        Параметры:
        - pull: Если True, модель будет загружена с Hugging Face Hub.
        """
        self.conf_Hyper = conf_Hyper
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf_Hyper.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.conf_Hyper.model_path, torch_dtype=torch.float16)
        self.model = self.model.to('cuda:0')
        self.params = {
                "max_length": self.conf_Hyper.max_length,
                "min_length": self.conf_Hyper.min_length,
                "num_return_sequences": self.conf_Hyper.num_return_sequences,
                "temperature": self.conf_Hyper.temperature,
                "top_k": self.conf_Hyper.top_k,
                "top_p": self.conf_Hyper.top_p,
                "do_sample": self.conf_Hyper.do_sample,
                "repetition_penalty": self.conf_Hyper.repetition_penalty,
                "length_penalty": self.conf_Hyper.length_penalty,
                "early_stopping": self.conf_Hyper.early_stopping,
                "num_beams": self.conf_Hyper.num_beams,
                "no_repeat_ngram_size": self.conf_Hyper.no_repeat_ngram_size          
        }

        # Установка токена заполнения
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Если pad_token_id == eos_token_id



    def generate(self, user_prompt: str, chunk:str='') -> str:
        """
        Метод для создания чата с моделью и генерации ответа на вопрос по ролям.

        Параметры:
        - system_prompt: str: Промпт для системы (инструкции).
        - user_prompt: str: Пользовательский промпт для LLM.
        - assistant_history: list: История ответов ассистента для поддержания контекста.
        - generation_kwargs: Дополнительные гиперпараметры для функции generate.

        Возвращает:
        - response: str: Сгенерированный ответ от модели.
        """
        # Формируем полный ввод для модели
        input_sequence = [f"System: {self.conf_Hyper.system_prompt}"] + \
                         [f"User: {user_prompt}"] + \
                         [f"Assistant: {self.conf_Hyper.assistant_prompt} + {chunk}"]

        # Объединяем все воедино
        full_input = "\n".join(input_sequence)

        # Подготавливаем текст для модели
        inputs = self.tokenizer(full_input, return_tensors='pt', padding=True, truncation=True).to('cuda:0')

        # Генерация ответа от модели с дополнительными гиперпараметрами
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **self.params)

        # Декодирование ответа
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


if __name__ == "__main__":

    conf = LLM_Hyper_Conf()
    model = LLM_model(conf)

    # Примеры пользовательских запросов
    user_prompt = "Расскажи мне анекдот."

        
    print(f"User: {user_prompt}")
    response = model.generate(user_prompt)
    print(f"Model: {response}\n")
    



