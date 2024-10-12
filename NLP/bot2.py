import Summarizer.summarizerTr2 as sum
import Retriever.retriever as retr
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

class Bot:
    """
    Класс чат-бота с использованием RAG
    """

    def __init__(self,pdf_path:str) -> None:
        """
        Конструктор класса Bot

        pdf_path: Путь до локальной языковой модели
        """
        self.conf = sum.LLM_Hyper_Conf()
        self.summarizer = sum.LLM_model(self.conf)
        self.retriever = retr.Retriever()
        self.pdf_path = pdf_path


    def proceed(self, query:str, faiss_path:str="/home/user1/QnA_bot_RZD/NLP/faiss_index.index", load_faiss: bool=True) -> str:
        """
        Метод для отправки вопроса чат-боту в консольной варианте

        query: Вопрос пользователя

        Return:str
        """
        if load_faiss:
            self.retriever.index = self.retriever.load_faiss_index(faiss_path)
            texts = self.retriever.texts
            filenames = self.retriever.filenames

        else:
            index, texts, filenames = self.retriever.create_faiss_index_from_pdf(self.pdf_path)
            self.retriever.index = index
            self.retriever.texts = texts
            self.retriever.filenames = filenames
            self.retriever.save_faiss_index(index, faiss_path)

        n_results = 5
        results = self.retriever.retriever(query, top_k=n_results)
        print(results)
        filenames_results = []
        for idx in range(len(texts)):
            if texts[idx] in (results["faiss"] + results["bm25"]):
                filenames_results.append(filenames[idx])
        filenames_results = set(filenames_results)
        output = self.summarizer.Generate(query, results["faiss"] + results["bm25"])

        return output + "\n Найдено в файлах:\n● " + '\n● '.join(filenames_results)


# pdf_path = "/home/user1/QnA_bot_RZD/Documents/base/" # Укажите путь к вашему PDF файлу
# bot = Bot(pdf_path)
# query = "Сколько длится ежегодный основной отпуск?"
# print("ОТВЕТ МОДЕЛИ:\n", bot.proceed(query, load_faiss=True))
    # "Сколько длится ежегодный основной отпуск?",
    # "Кто такой представитель работодателя?", 
    # "Сколько раз индексируется заработная плата?", 
    # "На основе какой статьи трудового кодекса Российской Федерации исчисляется размер среднечасового заработка?" 