import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from .utilsTr import LLM_Hyper_Conf


class LLM_model:
    """
    Класс для инициализации модели и обращения к ней
    """

    def __init__(self, conf_Hyper:LLM_Hyper_Conf) -> None:
        """
        Инициализация модели.
        
        Параметры:
        - conf_Hyper: Конфиг модели лежащий в utils.
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
        self.generation_config = GenerationConfig.from_pretrained(self.conf_Hyper.model_path)


    def Generate(self, user_prompt: str, chunk:str='') -> str:
        """
        Метод для создания чата с моделью и генерации ответа на вопрос по ролям.

        Параметры:
        - user_prompt: str: Пользовательский промпт для LLM.
        - chunk: str: Чанк релевантной информации

        Возвращает:
        - response: str: Сгенерированный ответ от модели.
        """


        prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": self.conf_Hyper.system_prompt
        }, {
            "role": "user",
            "content": user_prompt
        }, {
            "role": "assisstant",
            "content": str(self.conf_Hyper.assistant_prompt) + str(chunk)
        }
        ], tokenize=False, add_generation_prompt=True)


        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, generation_config=self.generation_config, max_new_tokens=128)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return response
    



