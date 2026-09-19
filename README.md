# 🎬 سیستم پیشنهاد فیلم

یک سیستم پیشنهاد فیلم ساخته‌شده با **Python** و **Streamlit** که به کاربر اجازه می‌دهد یک فیلم را جستجو کند و فیلم‌های مشابه آن را بر اساس محتوای فیلم دریافت کند.

## ✨ قابلیت‌ها

* 🔎 جستجوی فیلم بر اساس نام
* 🎯 پیشنهاد فیلم‌های مشابه
* ⭐ نمایش امتیاز فیلم
* 🗳️ نمایش تعداد رأی‌ها
* 📅 نمایش تاریخ انتشار
* 🎭 نمایش ژانرها
* 🖼️ نمایش پوستر فیلم
* 📖 نمایش توضیحات فیلم
* 🌙 حالت تاریک
* ☀️ حالت روشن
* ⚡ جستجوی سریع بین فیلم‌ها
* 🔙 امکان بازگشت به صفحات قبلی
* 🎯 باز کردن صفحه جزئیات فیلم‌های پیشنهادی
* 📦 استفاده از فرمت Parquet برای بارگذاری سریع‌تر دیتاست
* 🤖 استفاده از TF-IDF برای تبدیل اطلاعات متنی فیلم‌ها به بردارهای عددی
* ⚡ استفاده از `sparse-dot-topn` برای محاسبه سریع شباهت بین فیلم‌ها

## 🛠️ تکنولوژی‌های استفاده‌شده

* Python
* Streamlit
* Pandas
* NumPy
* SciPy
* Scikit-learn
* PyArrow
* Pillow
* sparse-dot-topn
* Jupyter Notebook

## 🧠 سیستم پیشنهاد فیلم

در این پروژه از الگوریتم **TF-IDF (Term Frequency-Inverse Document Frequency)** برای تبدیل اطلاعات متنی فیلم‌ها به بردارهای عددی استفاده شده است.

اطلاعات متنی مورد استفاده برای ساخت ویژگی‌های فیلم شامل:

* ژانرها (`genres`)
* کلمات کلیدی (`keywords`)
* توضیحات فیلم (`overview`)

این اطلاعات در یک ستون به نام `tags` ترکیب و سپس پاک‌سازی می‌شوند.

پس از آن، با استفاده از `TfidfVectorizer` یک ماتریس TF-IDF با حداکثر **10,000 ویژگی** ساخته می‌شود.

برای پیدا کردن فیلم‌های مشابه، شباهت کسینوسی بین بردار فیلم انتخاب‌شده و سایر فیلم‌ها محاسبه می‌شود و نزدیک‌ترین فیلم‌ها به کاربر نمایش داده می‌شوند.

## 📊 دیتاست

این پروژه از **TMDB Movie Dataset** استفاده می‌کند.

دیتاست اصلی به صورت CSV در اختیار برنامه قرار می‌گیرد:

```text
data/TMDB_movie_dataset_v11.csv
```

اسکریپت `convert_dataset.py` دیتاست را پردازش می‌کند و:

1. فیلم‌های بدون عنوان را حذف می‌کند.
2. ستون‌های متنی موردنیاز را آماده می‌کند.
3. ستون `tags` را از `genres`، `keywords` و `overview` ایجاد می‌کند.
4. متن را پاک‌سازی می‌کند.
5. دیتاست نهایی را به فرمت Parquet تبدیل می‌کند.

خروجی:

```text
data/movies.parquet
```

در نسخه فعلی پروژه، پس از پردازش، دیتاست شامل حدود **1.29 میلیون فیلم** است.

## 🤖 ساخت TF-IDF

پس از ساخت `movies.parquet`، اسکریپت زیر TF-IDF را ایجاد می‌کند:

```text
convert_tfidf.py
```

اجرای آن:

```powershell
python convert_tfidf.py
```

خروجی TF-IDF به صورت چند فایل ذخیره می‌شود:

```text
tfidf_data.npy
tfidf_indices.npy
tfidf_indptr.npy
tfidf_shape.txt
```

این فایل‌ها در زمان اجرای برنامه دوباره به یک Sparse Matrix تبدیل می‌شوند.

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
├── convert_dataset.py
├── convert_tfidf.py
├── requirements.txt
├── tfidf_shape.txt
├── .gitignore
└── README.md
```

> فایل‌های حجیم دیتاست و مدل به دلیل حجم زیاد در مخزن GitHub قرار نگرفته‌اند و توسط `.gitignore` نادیده گرفته می‌شوند.

## 📦 فایل‌های موردنیاز برای اجرای برنامه

برای اجرای کامل برنامه، فایل‌های زیر باید در پروژه وجود داشته باشند:

```text
data/movies.parquet

tfidf_data.npy
tfidf_indices.npy
tfidf_indptr.npy
tfidf_shape.txt
```

اگر این فایل‌ها هنوز ساخته نشده‌اند، ابتدا دیتاست اصلی را در مسیر زیر قرار دهید:

```text
data/TMDB_movie_dataset_v11.csv
```

سپس اجرا کنید:

```powershell
python convert_dataset.py
```

و بعد:

```powershell
python convert_tfidf.py
```

## 🚀 نصب و راه‌اندازی

ابتدا مخزن پروژه را Clone کنید:

```bash
git clone https://github.com/moeinghafari/movie-recommender.git
```

و وارد پوشه پروژه شوید:

```bash
cd movie-recommender
```

یک محیط مجازی ایجاد کنید:

```bash
python -m venv .venv
```

در Windows محیط مجازی را فعال کنید:

```powershell
.venv\Scripts\activate
```

سپس وابستگی‌های پروژه را نصب کنید:

```powershell
pip install -r requirements.txt
```

## ▶️ اجرای برنامه

پس از آماده بودن فایل‌های دیتاست و TF-IDF، برنامه را اجرا کنید:

```powershell
streamlit run app.py
```

سپس برنامه در مرورگر باز خواهد شد.

## 📓 Notebookها

### `01_data_exploration.ipynb`

برای بررسی و تحلیل اولیه دیتاست فیلم‌ها استفاده شده است.

### `02_movie_recommender_new.ipynb`

برای توسعه و آزمایش سیستم پیشنهاد فیلم و ساخت مدل TF-IDF استفاده شده است.

## ⚠️ فایل‌های حجیم

فایل‌های زیر به دلیل حجم زیاد در GitHub قرار نگرفته‌اند:

```text
*.csv
*.parquet
*.pkl
*.npy
```

این فایل‌ها توسط `.gitignore` از Git خارج شده‌اند.

## 👨‍💻 پروژه

GitHub:

https://github.com/moeinghafari/movie-recommender.git
