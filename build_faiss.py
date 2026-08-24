import joblib
import faiss
import numpy as np
import time


print("در حال بارگذاری TF-IDF...")

start = time.time()

tfidf_matrix = joblib.load(
    "tfidf_matrix.pkl"
)

print(
    f"TF-IDF بارگذاری شد: "
    f"{time.time() - start:.2f} ثانیه"
)


print("در حال تبدیل TF-IDF به float32...")

start = time.time()

tfidf_matrix = tfidf_matrix.astype(
    np.float32
)

print(
    f"تبدیل انجام شد: "
    f"{time.time() - start:.2f} ثانیه"
)


print("در حال ساخت FAISS Index...")

start = time.time()

index = faiss.IndexFlatIP(
    tfidf_matrix.shape[1]
)

index.add(
    tfidf_matrix
)

print(
    f"FAISS Index ساخته شد: "
    f"{time.time() - start:.2f} ثانیه"
)


print("تعداد فیلم‌های داخل Index:")

print(
    index.ntotal
)


print("در حال ذخیره Index...")

faiss.write_index(
    index,
    "faiss_index.bin"
)


print("✅ FAISS Index با موفقیت ذخیره شد!")

print(
    "حجم ویژگی‌ها:",
    tfidf_matrix.shape
)