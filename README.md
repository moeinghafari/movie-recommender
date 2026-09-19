# 🎬 سیستم پیشنهاد فیلم

یک سیستم پیشنهاد فیلم ساخته‌شده با استفاده از **Python** و **Streamlit**.

این برنامه به کاربر اجازه می‌دهد نام یک فیلم را جستجو کند و فیلم‌های مشابه آن را دریافت کند.

## ✨ قابلیت‌ها

* 🔎 جستجوی فیلم بر اساس نام
* 🎯 پیشنهاد فیلم‌های مشابه
* ⭐ نمایش امتیاز فیلم
* 🗳️ نمایش تعداد رأی‌ها
* 📅 نمایش تاریخ انتشار
* 🖼️ نمایش پوستر فیلم
* 📖 نمایش توضیحات فیلم
* 🌙 حالت تاریک
* ☀️ حالت روشن
* ⚡ جستجوی سریع فیلم‌های مشابه
* 💾 بارگذاری بهینه داده‌ها با استفاده از قابلیت Cache در Streamlit
* 📦 استفاده از فرمت Parquet برای بارگذاری سریع‌تر دیتاست
* 🤖 استفاده از TF-IDF برای محاسبه شباهت بین فیلم‌ها

## 🛠️ تکنولوژی‌های استفاده‌شده

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Joblib
* PyArrow

## 🧠 سیستم پیشنهاد فیلم

در این پروژه از الگوریتم **TF-IDF (Term Frequency-Inverse Document Frequency)** برای تبدیل اطلاعات متنی فیلم‌ها به بردارهای عددی استفاده شده است.

ماتریس Sparse حاصل از TF-IDF برای محاسبه میزان شباهت بین فیلم‌ها استفاده می‌شود.

برای پیدا کردن فیلم‌های مشابه، از **Cosine Similarity** استفاده شده و برنامه نزدیک‌ترین فیلم‌ها به فیلم انتخاب‌شده را نمایش می‌دهد.

## 📁 ساختار پروژه

```text
movie-recommender/
│
├── data/
│   └── movies.parquet
│
├── notebook/
│   ├── 01_data_exploration.ipynb
│   └── 02_movie_recommender_new.ipynb
│
├── app.py
├── build_faiss.py
├── convert_dataset.py
├── convert_tfidf.py
├── requirements.txt
├── tfidf_data.npy
├── tfidf_indices.npy
├── tfidf_indptr.npy
├── tfidf_shape.txt
├── .gitignore
└── README.md
```

> **نکته:** دیتاست و فایل‌های مدل TF-IDF به دلیل حجم زیاد، داخل مخزن GitHub قرار نگرفته‌اند.

## 📦 فایل‌های موردنیاز برنامه

برنامه به فایل‌های زیر نیاز دارد:

```text
data/movies.parquet

tfidf_data.npy
tfidf_indices.npy
tfidf_indptr.npy
tfidf_shape.txt
```

این فایل‌ها باید در ساختار زیر قرار داشته باشند:

```text
movie-recommender/
│
├── data/
│   └── movies.parquet
│
├── tfidf_data.npy
├── tfidf_indices.npy
├── tfidf_indptr.npy
└── tfidf_shape.txt
```

برنامه دیتاست فیلم‌ها را از مسیر زیر بارگذاری می‌کند:

```text
data/movies.parquet
```

و ماتریس Sparse مربوط به TF-IDF را از فایل‌های زیر بازسازی می‌کند:

```text
tfidf_data.npy
tfidf_indices.npy
tfidf_indptr.npy
tfidf_shape.txt
```

## 🚀 نصب و راه‌اندازی

ابتدا مخزن پروژه را Clone کنید:

```bash
git clone https://github.com/moeinghafari/movie-recommender.git
```

سپس وارد پوشه پروژه شوید:

```bash
cd movie-recommender
```

یک Virtual Environment ایجاد کنید:

```bash
python -m venv .venv
```

در ویندوز محیط مجازی را فعال کنید:

```powershell
.venv\Scripts\activate
```

سپس کتابخانه‌های موردنیاز را نصب کنید:

```bash
pip install -r requirements.txt
```

## ▶️ اجرای برنامه

بعد از قرار دادن فایل‌های موردنیاز در مسیرهای مشخص‌شده، برنامه را با دستور زیر اجرا کنید:

```bash
streamlit run app.py
```

بعد از اجرای دستور، برنامه در مرورگر باز خواهد شد.

## 📊 دیتاست

این پروژه از **TMDB Movie Dataset** استفاده می‌کند.

برای تبدیل فایل CSV به فرمت Parquet می‌توان از فایل زیر استفاده کرد:

```text
convert_dataset.py
```

این اسکریپت انتظار دارد فایل اصلی دیتاست در مسیر زیر قرار داشته باشد:

```text
data/TMDB_movie_dataset_v11.csv
```

سپس فایل زیر را ایجاد می‌کند:

```text
data/movies.parquet
```

به دلیل حجم زیاد دیتاست، فایل اصلی CSV در مخزن GitHub قرار نگرفته است.

## 📓 نوت‌بوک‌ها

این پروژه شامل نوت‌بوک‌های زیر است که در مراحل توسعه و تحلیل داده استفاده شده‌اند:

### `01_data_exploration.ipynb`

برای بررسی و تحلیل اولیه دیتاست فیلم‌ها استفاده شده است.

### `02_movie_recommender_new.ipynb`

برای توسعه و آزمایش سیستم پیشنهاد فیلم و ساخت نمایش عددی TF-IDF استفاده شده است.

## ⚠️ فایل‌های حجیم

فایل‌های زیر عمداً توسط Git نادیده گرفته می‌شوند:

```text
*.csv
*.parquet
*.pkl
*.npy
```

این کار باعث می‌شود مخزن GitHub حجم مناسبی داشته باشد و دیتاست‌های بزرگ و فایل‌های مدل وارد مخزن نشوند.

## 👨‍💻 پروژه

مخزن GitHub پروژه:

https://github.com/moeinghafari/movie-recommender
