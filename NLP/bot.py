import Summarizer.summarizer as sum
import Retriever.retriever as retr
import Vectorizer.vectorizer as vect
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

class Bot:

    def __init__(self,pdf_path) -> None:
        self.vectorizer = vect.Vectorizer(512)
        self.conf1 = sum.LLM_Hardw_Conf()
        self.conf2 = sum.LLM_Hyper_Conf()
        self.summarizer = sum.LLM_model(self.conf1, self.conf2, pull=False)
        self.retriever = retr.Retriever()
        self.pdf_path = pdf_path


    def proceed(self,query):

        index, texts = self.vectorizer.create_faiss_index_from_pdf(self.pdf_path)
        self.retriever.index = index
        self.retriever.texts = texts

        index_file_path = "faiss_index.index"
        self.vectorizer.save_faiss_index(index, index_file_path)
        print(f"Индекс FAISS сохранен по пути: {index_file_path}")
 
        n_results = 5
        results = self.retriever.retriever(query, top_k=n_results)
        print(results)

        output = self.summarizer.generate(query, results['faiss'][:n_results])

        if self.conf2.stream:

            for item in output:
                try:
                    print(item['choices'][0]['delta']['content'], end='')
                except Exception:
                    continue
        else:
            print(output['choices'][0]['message']['content'])



pdf_path = "/home/user1/QnA_rzd/Documents/base/Коллективный договор.pdf" # Укажите путь к вашему PDF файлу
bot = Bot(pdf_path)
query = "Когда начал действовать данный договор и когда закончит действовать?"
bot.proceed(query)