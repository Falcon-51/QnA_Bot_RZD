import gradio as gr
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from bot2 import Bot
torch.cuda.empty_cache()
import gc
gc.collect()

UI_TITLE = "Чат-бот корпоративного университета РЖД"
UI_DESCRIPTION = "Чат-бот может ответить на ваши вопросы по нормативным документов из внутренней базы знаний корпоративного университета РЖД. \
База документов, по которой осуществляется поиск, была получена на основе следующего ресурса: ['РЖД. Документы | Компания'](https://company.rzd.ru/ru/9353)."
UI_ARTICLE = ''
QUESTION_EXAMPLES = [
    "Сколько длится ежегодный основной отпуск?",
    "Кто такой представитель работодателя?", 
    "На основе какой статьи трудового кодекса Российской Федерации исчисляется размер среднечасового заработка?" 
    ]
UNDO_BTN_NAME = 'Удалить предыдущий вопрос/ответ'
CLEAR_BTN_NAME = 'Очистить весь чат'
QUESTION_FIELD_PLACEHOLDER = "Какой у вас вопрос?"
SUBMIT_BTN = 'Отправить'
BOT_AVATAR, USER_AVATAR = [
    '/home/user1/QnA_rzd/NLP/data/bot_avatar.jpg',
    '/home/user1/QnA_rzd/NLP/data/user_avatar.png']



class StopOnTokens(StoppingCriteria):
    """
    Класс для отслеживания служебного токена-остановки
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    

class QnA:
    """
    Класс для инференса чат-бота
    """

    def __init__(self, pdf_path:str="/home/user1/QnA_rzd/Documents/base/", faiss_path:str="/home/user1/QnA_rzd/NLP/faiss_index.index", load_faiss:bool=True):

        """
        Конструктор класса чат-бота

        pdf-path: директория с документами
        faiss_path: путь до файла с БД Faiss
        load_faiss: флаг загрузки файла БД Faiss
        """
        self.pdf_path = pdf_path
        self.llm = Bot(pdf_path)

        if load_faiss:
            self.llm.retriever.index = self.llm.retriever.load_faiss_index(faiss_path)
            texts = self.llm.retriever.texts
            filenames = self.llm.retriever.filenames

        else:
            index, texts, filenames = self.llm.retriever.create_faiss_index_from_pdf(self.pdf_path)
            self.llm.retriever.index = index
            self.llm.retriever.texts = texts
            self.llm.retriever.filenames = filenames
            self.llm.retriever.save_faiss_index(index, faiss_path)


            #index_file_path = "faiss_index.index"
            self.llm.retriever.save_faiss_index(index, faiss_path)



    def predict(self, message:str, history:str):
        """
        Метод для запуска приложения Gradio

        message: запрос пользователя
        history: история общения
        """

        n_results = 5
        
        stop = StopOnTokens()

        history_transformer_format = history + [[message, ""]]
        history = message
        messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]]) for item in history_transformer_format])

        messages = message

        results = self.llm.retriever.retriever(messages, top_k=n_results)
        streamer = TextIteratorStreamer(self.llm.summarizer.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

        prompt = self.llm.summarizer.tokenizer.apply_chat_template([{
                    "role": "system",
                    "content": "Ты бот помощник"
                }, {
                    "role": "user",
                    "content": messages
                }, {
                    "role": "assisstant",
                    "content": f'Отвечай на основе этого текста: {results["faiss"] + results["bm25"]}'
                }
                ], tokenize=False, add_generation_prompt=True)


        data = self.llm.summarizer.tokenizer(prompt, return_tensors="pt").to("cuda")



        generate_kwargs = dict(
            data,
            streamer=streamer,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=0.7,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
        )



        t = Thread(target=self.llm.summarizer.model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message  = ""
        for new_token in streamer:
            if new_token != '<':
                partial_message += new_token
                yield partial_message



if __name__ == "__main__":

    model = QnA()
    gr.ChatInterface(model.predict,
                     title=UI_TITLE,
                     chatbot=gr.Chatbot(
                     height=400, 
                     show_copy_button=True,
                     avatar_images=[USER_AVATAR, BOT_AVATAR]),
                     textbox=gr.Textbox(placeholder=QUESTION_FIELD_PLACEHOLDER, container=False, scale=7),
                     description=UI_DESCRIPTION,
                     theme=gr.themes.Default(),
                     examples=QUESTION_EXAMPLES,
                     retry_btn=None,
                     undo_btn=UNDO_BTN_NAME,
                     clear_btn=CLEAR_BTN_NAME,
                     submit_btn=SUBMIT_BTN).launch(server_name='0.0.0.0',share=True)