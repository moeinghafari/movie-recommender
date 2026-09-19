import pandas as pd
import re
import time

print("در حال بارگذاری دیتاست اصلی...")

start = time.time()

movies = pd.read_csv(
    "data/TMDB_movie_dataset_v11.csv",
    low_memory=False
)

print(
    f"دیتاست بارگذاری شد: "
    f"{time.time() - start:.2f} ثانیه"
)

print("تعداد اولیه فیلم‌ها:", len(movies))


# حذف فیلم‌هایی که عنوان ندارند
movies_clean = movies[
    movies["title"].notna()
    & movies["title"].astype(str).str.strip().ne("")
].copy()


# آماده‌سازی ستون‌های متنی
for column in ["genres", "keywords", "overview"]:
    movies_clean[column] = (
        movies_clean[column]
        .fillna("")
        .astype(str)
    )


# تمیز کردن متن
def clean_text(value):
    value = "" if pd.isna(value) else str(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# ساخت tags
movies_clean["tags"] = (
    movies_clean["genres"]
    + " "
    + movies_clean["keywords"]
    + " "
    + movies_clean["overview"]
).map(clean_text)


# حذف فیلم‌هایی که tags خالی دارند
movies_clean = movies_clean[
    movies_clean["tags"].str.len() > 0
].reset_index(drop=True)


print("تعداد فیلم‌های نهایی:", len(movies_clean))

print("آیا ستون tags وجود دارد؟", "tags" in movies_clean.columns)


# ذخیره Parquet
print("در حال ذخیره movies.parquet...")

movies_clean.to_parquet(
    "data/movies.parquet",
    index=False
)

print("تبدیل با موفقیت انجام شد.")