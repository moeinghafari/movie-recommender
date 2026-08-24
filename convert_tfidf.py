import joblib
import numpy as np
import time


print("در حال بارگذاری TF-IDF...")

start = time.time()

tfidf_matrix = joblib.load(
    "tfidf_matrix.pkl"
)

print(
    f"بارگذاری انجام شد: "
    f"{time.time() - start:.2f} ثانیه"
)


print("در حال ذخیره‌سازی با فرمت سریع...")


np.save(
    "tfidf_data.npy",
    tfidf_matrix.data
)

np.save(
    "tfidf_indices.npy",
    tfidf_matrix.indices
)

np.save(
    "tfidf_indptr.npy",
    tfidf_matrix.indptr
)


with open(
    "tfidf_shape.txt",
    "w"
) as f:

    f.write(
        f"{tfidf_matrix.shape[0]},{tfidf_matrix.shape[1]}"
    )


print("تبدیل با موفقیت انجام شد.")

print(
    "Shape:",
    tfidf_matrix.shape
)