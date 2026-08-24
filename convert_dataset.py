import pandas as pd
import time

start = time.time()

movies = pd.read_csv(
    "data/TMDB_movie_dataset_v11.csv",
    low_memory=False
)

movies.to_parquet(
    "data/movies.parquet",
    index=False
)

end = time.time()

print(
    f"تبدیل با موفقیت انجام شد: "
    f"{end - start:.2f} ثانیه"
)