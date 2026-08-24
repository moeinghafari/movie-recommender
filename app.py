import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors


# =========================================================
# تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)
# =========================
# حالت شب و روز
# =========================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


col1, col2 = st.columns([10, 1])

with col2:

    if st.session_state.dark_mode:

        if st.button(
            "☀️",
            key="theme_button"
        ):

            st.session_state.dark_mode = False
            st.rerun()

    else:

        if st.button(
            "🌙",
            key="theme_button"
        ):

            st.session_state.dark_mode = True
            st.rerun()


# =========================================================
# CSS - طراحی رابط کاربری
# =========================================================

# =========================
# طراحی ظاهری
# =========================

if st.session_state.dark_mode:

    background_color = "#0E1117"
    text_color = "#FFFFFF"
    card_color = "#161B22"
    border_color = "#30363D"

else:

    background_color = "#FFFFFF"
    text_color = "#111111"
    card_color = "#F8F9FA"
    border_color = "#DDDDDD"


st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}

    .main-title {{
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        margin-bottom: 5px;
        color: {text_color};
    }}

    .subtitle {{
        text-align: center;
        color: #888888;
        font-size: 18px;
        margin-bottom: 30px;
    }}

    .movie-card {{
        background-color: {card_color};
        padding: 15px;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-bottom: 15px;
    }}

    .movie-title {{
        font-size: 22px;
        font-weight: bold;
        color: {text_color};
    }}

    .movie-info {{
        font-size: 16px;
        margin-top: 8px;
        color: {text_color};
    }}

    .similar-button button {{
        background-color: #0066FF !important;
        color: white !important;
        border: 1px solid #0066FF !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }}

    .similar-button button:hover {{
        background-color: #0052CC !important;
    }}

    .watch-button button {{
        background-color: #000000 !important;
        color: white !important;
        border: 1px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }}

    .watch-button button:hover {{
        background-color: #222222 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# بارگذاری دیتاست
# =========================================================

@st.cache_data
def load_movies():

    return pd.read_csv(
        "data/TMDB_movie_dataset_v11.csv",
        low_memory=False
    )


# =========================================================
# بارگذاری مدل
# =========================================================

@st.cache_resource
def load_model():

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    tfidf_matrix = joblib.load(
        "tfidf_matrix.pkl"
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=21,
        n_jobs=-1
    )

    model.fit(
        tfidf_matrix
    )

    return tfidf, tfidf_matrix, model


# =========================================================
# اجرای بارگذاری
# =========================================================

movies = load_movies()

tfidf, tfidf_matrix, model = load_model()


# =========================================================
# Session State
# =========================================================

if "selected_movie_index" not in st.session_state:

    st.session_state.selected_movie_index = None


if "recommendations" not in st.session_state:

    st.session_state.recommendations = None


if "selected_search_movie" not in st.session_state:

    st.session_state.selected_search_movie = None


# =========================================================
# تابع نمایش جزئیات فیلم
# =========================================================

def show_movie_details(movie_index):

    movie = movies.iloc[movie_index]

    st.divider()

    col1, col2 = st.columns(
        [1, 3]
    )

    # -----------------------------------------------------
    # پوستر
    # -----------------------------------------------------

    with col1:

        if pd.notna(
            movie.get("poster_path")
        ):

            st.image(
                "https://image.tmdb.org/t/p/w500"
                + str(movie["poster_path"]),
                width=300
            )

        else:

            st.info(
                "پوستر این فیلم موجود نیست."
            )

    # -----------------------------------------------------
    # اطلاعات فیلم
    # -----------------------------------------------------

    with col2:

        st.markdown(
            f"## 🎬 {movie['title']}"
        )

        st.write(
            f"⭐ **امتیاز:** {movie['vote_average']}"
        )

        st.write(
            f"🗳 **تعداد رأی:** "
            f"{movie['vote_count']:,}"
        )

        st.write(
            f"📅 **تاریخ انتشار:** "
            f"{movie['release_date']}"
        )

        if pd.notna(
            movie.get("genres")
        ):

            st.write(
                f"🎭 **ژانر:** "
                f"{movie['genres']}"
            )

        st.markdown(
            "### 📖 درباره فیلم"
        )

        if pd.notna(
            movie.get("overview")
        ):

            st.write(
                movie["overview"]
            )

        else:

            st.write(
                "توضیحاتی برای این فیلم وجود ندارد."
            )


# =========================================================
# هدر اصلی
# =========================================================

st.markdown(
    '<div class="main-title">🎬 Movie Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        یک فیلم را جستجو کنید و فیلم‌های مشابه آن را پیدا کنید.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# صفحه جزئیات فیلم
# =========================================================

if st.session_state.selected_movie_index is not None:

    selected_index = (
        st.session_state.selected_movie_index
    )

    selected_movie_data = movies.iloc[
        selected_index
    ]

    st.markdown(
        f"# 🎬 {selected_movie_data['title']}"
    )

    # -----------------------------------------------------
    # دکمه بازگشت
    # -----------------------------------------------------

    if st.button(
        "⬅️ بازگشت به صفحه جستجو"
    ):

        st.session_state.selected_movie_index = None

        st.session_state.recommendations = None

        st.session_state.selected_search_movie = None

        st.rerun()

    # -----------------------------------------------------
    # جزئیات فیلم
    # -----------------------------------------------------

    show_movie_details(
        selected_index
    )


# =========================================================
# صفحه اصلی جستجو
# =========================================================

else:

    # -----------------------------------------------------
    # جستجوی فیلم
    # -----------------------------------------------------

    search = st.text_input(
        "🔎 جستجوی نام فیلم",
        placeholder="مثلاً: Inception"
    )


    # =====================================================
    # نمایش نتایج جستجو
    # =====================================================

    if search:

        results = movies[
            movies["title"]
            .fillna("")
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ]


        st.markdown(
            f"### 🔎 نتایج جستجو برای: `{search}`"
        )


        if results.empty:

            st.warning(
                "❌ فیلمی با این نام پیدا نشد."
            )


        else:

            st.success(
                f"✅ {len(results):,} فیلم پیدا شد."
            )


            # -------------------------------------------------
            # انتخاب فیلم
            # -------------------------------------------------

            selected_movie = st.selectbox(
                "🎬 فیلم موردنظر خود را انتخاب کنید",
                results["title"].tolist()
            )


            movie_index = results[
                results["title"]
                == selected_movie
            ].index[0]


            movie = movies.loc[
                movie_index
            ]


            st.divider()


            # =================================================
            # نمایش فیلم انتخاب شده
            # =================================================

            st.markdown(
                "### 🎥 فیلم انتخاب شده"
            )


            col1, col2 = st.columns(
                [1, 3]
            )


            # -------------------------------------------------
            # پوستر
            # -------------------------------------------------

            with col1:

                if pd.notna(
                    movie.get("poster_path")
                ):

                    st.image(
                        "https://image.tmdb.org/t/p/w500"
                        + str(
                            movie["poster_path"]
                        ),
                        width=280
                    )

                else:

                    st.info(
                        "پوستر موجود نیست."
                    )


            # -------------------------------------------------
            # اطلاعات
            # -------------------------------------------------

            with col2:

                st.markdown(
                    f"## 🎬 {movie['title']}"
                )

                st.write(
                    f"⭐ **امتیاز:** "
                    f"{movie['vote_average']}"
                )

                st.write(
                    f"🗳 **تعداد رأی:** "
                    f"{movie['vote_count']:,}"
                )

                st.write(
                    f"📅 **تاریخ انتشار:** "
                    f"{movie['release_date']}"
                )

                if pd.notna(
                    movie.get("genres")
                ):

                    st.write(
                        f"🎭 **ژانر:** "
                        f"{movie['genres']}"
                    )

                if pd.notna(
                    movie.get("overview")
                ):

                    st.write(
                        movie["overview"]
                    )

                else:

                    st.write(
                        "توضیحاتی برای این فیلم وجود ندارد."
                    )


            st.divider()


            # =================================================
            # تعداد پیشنهادها
            # =================================================

            st.markdown(
                "### 🎯 تنظیمات پیشنهاد"
            )


            number = st.slider(
                "تعداد فیلم‌های مشابه",
                min_value=5,
                max_value=20,
                value=10
            )


            # =================================================
            # دکمه پیشنهاد فیلم
            # =================================================

            if st.button(
                "🎯 پیشنهاد فیلم‌های مشابه",
                use_container_width=True
            ):

                with st.spinner(
                    "🔍 در حال پیدا کردن فیلم‌های مشابه..."
                ):

                    distances, indexes = (
                        model.kneighbors(
                            tfidf_matrix[
                                movie_index
                            ],
                            n_neighbors=number + 1
                        )
                    )


                    recommendations = []


                    # -------------------------------------------------
                    # ساخت لیست پیشنهادها
                    # -------------------------------------------------

                    for distance, index in zip(
                        distances[0],
                        indexes[0]
                    ):

                        index = int(index)


                        # حذف خود فیلم
                        if index == movie_index:

                            continue


                        recommendations.append({

                            "index": index,

                            "title":
                                movies.iloc[
                                    index
                                ]["title"],

                            "release_date":
                                movies.iloc[
                                    index
                                ]["release_date"],

                            "rating":
                                movies.iloc[
                                    index
                                ]["vote_average"],

                            "similarity":
                                round(
                                    (
                                        1 - float(
                                            distance
                                        )
                                    ) * 100,
                                    2
                                )

                        })


                        if len(
                            recommendations
                        ) >= number:

                            break


                # -------------------------------------------------
                # ذخیره در Session State
                # -------------------------------------------------

                st.session_state.recommendations = (
                    recommendations
                )

                st.session_state.selected_search_movie = (
                    selected_movie
                )


            # =================================================
            # نمایش پیشنهادها
            # =================================================

            if (
                st.session_state.recommendations
                is not None
            ):

                st.divider()


                st.markdown(
                    f"## 🎬 فیلم‌های مشابه "
                    f"`{st.session_state.selected_search_movie}`"
                )


                st.write(
                    "فیلم‌هایی که بر اساس محتوای خود به فیلم انتخاب‌شده نزدیک‌تر هستند:"
                )


                # -------------------------------------------------
                # نمایش هر فیلم
                # -------------------------------------------------

                for recommendation in (
                    st.session_state.recommendations
                ):

                    recommendation_index = int(
                        recommendation["index"]
                    )


                    recommendation_movie = (
                        movies.iloc[
                            recommendation_index
                        ]
                    )


                    st.markdown(
                        '<div class="movie-card">',
                        unsafe_allow_html=True
                    )


                    col1, col2, col3 = st.columns(
                        [1, 5, 2]
                    )


                    # =================================================
                    # پوستر
                    # =================================================

                    with col1:

                        if pd.notna(
                            recommendation_movie.get(
                                "poster_path"
                            )
                        ):

                            st.image(
                                "https://image.tmdb.org/t/p/w300"
                                + str(
                                    recommendation_movie[
                                        "poster_path"
                                    ]
                                )
                            )

                        else:

                            st.info(
                                "بدون پوستر"
                            )


                    # =================================================
                    # اطلاعات
                    # =================================================

                    with col2:

                        st.markdown(
                            f"### "
                            f"{recommendation_movie['title']}"
                        )

                        st.write(
                            f"📅 تاریخ انتشار: "
                            f"{recommendation_movie['release_date']}"
                        )

                        st.write(
                            f"⭐ امتیاز: "
                            f"{recommendation_movie['vote_average']}"
                        )

                        if pd.notna(
                            recommendation_movie.get(
                                "genres"
                            )
                        ):

                            st.write(
                                f"🎭 ژانر: "
                                f"{recommendation_movie['genres']}"
                            )


                    # =================================================
                    # شباهت + دکمه
                    # =================================================

                    with col3:

                        st.markdown(
                            f"""
                            <div class="similarity">
                                🎯 شباهت<br>
                                {recommendation['similarity']}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        st.write("")


                        if st.button(
                            "👁️ مشاهده فیلم",
                            key=(
                                f"movie_"
                                f"{recommendation_index}"
                            ),
                            use_container_width=True
                        ):

                            st.session_state.selected_movie_index = (
                                recommendation_index
                            )

                            st.session_state.recommendations = (
                                None
                            )

                            st.session_state.selected_search_movie = (
                                None
                            )

                            st.rerun()


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )