import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import pdfplumber
import os
import gc

class Retriever:
    """
    Класс ретривер для векторизации данных и сохранения в векторную БД
    """

    def __init__(self) -> None:
        """
        Инициализация модели векторизации
        """
        self.index = None
        self.texts = []
        self.filenames = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "/home/user1/QnA_Bot_RZD/NLP/llm_models/models--NousResearch--Llama-3.2-1B/snapshots/a9745ffc3556f145a830ac0c203509ba860582a3"
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'unk_token': '[UNK]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        #self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').to(self.device)
        


    def get_embeddings(self, texts: list):
        """
        Принимает на вход texts - список
        Возвращает эмбедденги размера [N, 2048]
                Метод для создания чата с моделью и генерации ответа на вопрос по ролям.

        Параметры:
        - texts: list: Текста для векторизации.

        Возвращает:
        - embeddings: np.array: Векторное представление текста.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            #outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()
        # outputs = outputs.hidden_states[-1]
        # embeddings = outputs.mean(dim=1).cpu().detach().numpy()
        #embeddings = self.model.encode(texts, convert_to_tensor=True).cpu().detach().numpy()
        return embeddings



    def create_faiss_index_from_pdf(self, pdf_path: str, chunk_size: int=256) -> tuple:
        """
        Разбивает PDF документы на чанки, преобразует в вектор и убирает в бд
        Аргументы:
        pdf_path: str - путь до pdf файла
        chunk_size: int - размер (в символах) одного чанка текста
        Возвращает:
        index: faiss index, texts: str
        """
        # Загружаем модель для создания эмбеддингов
        all_embeddings = []
        all_texts = []
        all_filenames = []

        # Проход по файлам в директории
        for filename in os.listdir(pdf_path):
            if filename.endswith('.pdf'):
                var = os.path.join(pdf_path, filename)


                with pdfplumber.open(var) as pdf:
                    # Чтение PDF файла и создание чанков
                    texts = []
                    
                    for page in pdf.pages:
                        text = page.extract_text().replace("\n", " ")
                        # Разделяем текст на чанки
                        for i in range(0, len(text), chunk_size):
                            chunk = text[i:i + chunk_size].strip()
                            if chunk:  # Добавляем только непустые чанки
                                texts.append(chunk)
                


                # Создание эмбеддингов для каждого чанка
                embeddings = self.get_embeddings(texts)

                all_embeddings.extend(embeddings)
                all_texts.extend(texts)
                all_filenames.extend([filename] * len(texts))

                # Создание FAISS индекса
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)  # Используем L2 расстояние
                index.add(np.array(embeddings, dtype=np.float32))  # Добавляем эмбеддинги в индекс

        return index, all_texts, all_filenames



    def save_faiss_index(self, index, file_path:str) -> None:
        """
        Сохранение базы Faiss
        """
        with open(file_path + ".txt", "w") as f:
            for line in self.texts:
                f.write(str(line) + "\n")

        with open(file_path + '_documents.txt', "w") as f:
            for document in self.filenames:
                f.write(str(document) + "\n")

        faiss.write_index(index, file_path)


    def load_faiss_index(self, file_path:str):
        """
        Загрузки из базы Faiss
        """
        with open(file_path+".txt", "r") as f:
            self.texts = f.readlines()

        with open(file_path+"_documents.txt", "r") as f:
            self.filenames = f.readlines()

        return faiss.read_index(file_path)
    

    def retriever(self, query:str, top_k:int=5):
        """
        Выбирает top_k наиболее релевантных документов для запроса query
        Аргументы:
            query: str - запрос пользователя
            top_k: int - количество наиболее релевантных документов
        Возвращает:
            Словарь с ключами bm25, faiss. {bm25: топ 5 документов, faiss: топ 5 документов}
        """
        if not self.index or not self.texts:
            raise ValueError("Индекс не загружен. Пожалуйста, создайте или загрузите индекс.")

        # Подготовка текстов для BM25
        tokenized_texts = [text.split() for text in self.texts]
        bm25 = BM25Okapi(tokenized_texts)

        # Получаем релевантные документы на основе BM25
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Получаем индексы топ-k релевантных результатов
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        # Получаем соответствующие тексты
        relevant_texts_bm25 = [self.texts[i] for i in top_indices]

        # Теперь используем FAISS для поиска наиболее близких векторов
        query_embedding = self.get_embeddings([query])

        # Поиск в FAISS
        D, I = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)

        # Получаем текстовые результаты на основе индексов
        relevant_texts_faiss = [self.texts[i] for i in I[0]]

        # Объединяем и возвращаем результаты
        return {
            "bm25": relevant_texts_bm25,
            "faiss": relevant_texts_faiss
        }

