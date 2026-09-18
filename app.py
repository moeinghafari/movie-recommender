import streamlit as st
import pandas as pd
import time
import numpy as np

from scipy.sparse import csr_matrix
from sparse_dot_topn import sp_matmul_topn


# =========================================================
# تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="سیستم پیشنهاد فیلم",
    page_icon="🎬",
    layout="wide"
)



# =========================================================
# Session State
# =========================================================

# =========================================================
# Session State
# =========================================================

# =========================================================
# Session State
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "selected_movie_index" not in st.session_state:
    st.session_state.selected_movie_index = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "selected_search_movie" not in st.session_state:
    st.session_state.selected_search_movie = None

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "selected_movie_title" not in st.session_state:
    st.session_state.selected_movie_title = None

if "recommendation_number" not in st.session_state:
    st.session_state.recommendation_number = 10
    
if "movie_search" not in st.session_state:
    st.session_state.movie_search = ""

if "recommendation_slider" not in st.session_state:
    st.session_state.recommendation_slider = 10
    
#vg============================================
# رنگ‌بندی حالت شب و روز
# =========================================================

if st.session_state.dark_mode:

    background_color = "#050B18"
    text_color = "#FFFFFF"
    secondary_text = "#CBD5E1"
    card_color = "#0C1425"
    input_color = "#0D1729"
    border_color = "#26344D"

else:

    background_color = "#F4F7FB"
    text_color = "#111827"
    secondary_text = "#475569"
    card_color = "#FFFFFF"
    input_color = "#FFFFFF"
    border_color = "#D5DCE8"


# =========================================================
# CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       تنظیم کلی
    ===================================================== */

    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}

    .block-container {{
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}

    /* تمام صفحه RTL */

    .stApp,
    .stApp * {{
        direction: rtl;
    }}

    /* متن‌ها */

    p,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    label,
    .stMarkdown {{
        text-align: right !important;
    }}


    /* =====================================================
       عنوان اصلی
    ===================================================== */

    .main-title {{
        text-align: center !important;
        direction: rtl !important;

        font-size: 54px;
        font-weight: 900;

        margin-top: 5px;
        margin-bottom: 5px;

        color: {text_color};

        letter-spacing: -1px;
    }}

    .main-subtitle {{
        text-align: center !important;
        direction: rtl !important;

        font-size: 19px;

        color: {secondary_text};

        margin-bottom: 30px;
    }}


    /* =====================================================
       دکمه شب و روز
    ===================================================== */

    .theme-button {{
        direction: rtl;
    }}

    .theme-button button {{
        border-radius: 10px !important;

        min-height: 42px !important;

        font-weight: bold !important;

        background-color: {card_color} !important;

        border: 1px solid {border_color} !important;

        color: {text_color} !important;
    }}


    /* =====================================================
       Search Box
    ===================================================== */

    [data-testid="stTextInput"] {{
        direction: rtl !important;

        margin-top: 5px;
        margin-bottom: 20px;
    }}

    [data-testid="stTextInput"] label {{
        direction: rtl !important;

        text-align: right !important;

        width: 100%;

        font-size: 17px;

        font-weight: bold;

        color: {text_color} !important;
    }}

    [data-testid="stTextInput"] input {{
        direction: rtl !important;

        text-align: right !important;

        background-color: {input_color} !important;

        color: {text_color} !important;

        border: 1px solid {border_color} !important;

        border-radius: 12px !important;

        min-height: 48px !important;

        font-size: 17px !important;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        direction: rtl !important;

        text-align: right !important;

        color: #94A3B8 !important;
    }}


    /* =====================================================
       Selectbox
    ===================================================== */

    [data-testid="stSelectbox"] {{
        direction: rtl !important;
    }}

    [data-testid="stSelectbox"] label {{
        direction: rtl !important;

        text-align: right !important;

        width: 100%;

        font-weight: bold;

        color: {text_color} !important;
    }}

    [data-testid="stSelectbox"] > div {{
        direction: rtl !important;
    }}


    /* =====================================================
       Slider
    ===================================================== */

    [data-testid="stSlider"] {{
        direction: rtl !important;
    }}

    [data-testid="stSlider"] label {{
        direction: rtl !important;

        text-align: right !important;

        width: 100%;

        font-weight: bold;

        color: {text_color} !important;
    }}


    /* =====================================================
       کارت فیلم اصلی
    ===================================================== */

    .main-movie-card {{
        background-color: {card_color};

        border: 1px solid {border_color};

        border-radius: 18px;

        padding: 25px;

        margin-top: 20px;

        margin-bottom: 30px;
    }}

    .main-movie-title {{
        font-size: 34px;

        font-weight: 800;

        color: {text_color};

        text-align: right;

        direction: rtl;

        margin-bottom: 20px;
    }}

    .movie-description {{
        color: {secondary_text};

        font-size: 16px;

        line-height: 2;

        text-align: right;

        direction: rtl;

        margin-top: 15px;
    }}


    /* =====================================================
       اطلاعات فیلم
    ===================================================== */

    .info-box {{
        display: inline-block;

        background-color: {input_color};

        border: 1px solid {border_color};

        border-radius: 10px;

        padding: 10px 14px;

        margin-left: 7px;

        margin-bottom: 8px;

        color: {text_color};

        font-size: 15px;

        direction: rtl;
    }}


    /* =====================================================
       عنوان بخش پیشنهادها
    ===================================================== */

    .recommendation-section-title {{
        font-size: 30px;

        font-weight: 800;

        color: {text_color};

        text-align: right;

        direction: rtl;

        margin-top: 25px;

        margin-bottom: 20px;
    }}


    /* =====================================================
       کارت پیشنهاد
    ===================================================== */

    .recommendation-card {{
        background-color: {card_color};

        border: 1px solid {border_color};

        border-radius: 15px;

        padding: 12px;

        margin-bottom: 15px;


        direction: rtl;
    }}

    .recommendation-title {{
        color: {text_color};

        font-size: 18px;

        font-weight: bold;

        text-align: right;

        direction: rtl;

        margin-top: 10px;

        min-height: 52px;
    }}

    .recommendation-info {{
        color: {secondary_text};

        font-size: 14px;

        text-align: right;

        direction: rtl;

        margin-top: 7px;
    }}

    .similarity {{
        color: #22C55E;

        font-size: 16px;

        font-weight: bold;

        text-align: right;

        direction: rtl;

        margin-top: 10px;

        margin-bottom: 10px;
    }}


    /* =====================================================
       دکمه اصلی پیشنهاد
    ===================================================== */

    .recommend-button button {{
        background: linear-gradient(
            135deg,
            #2563EB,
            #1D4ED8
        ) !important;

        color: white !important;

        border: none !important;

        border-radius: 11px !important;

        font-size: 17px !important;

        font-weight: bold !important;

        min-height: 48px !important;
    }}

    .recommend-button button:hover {{
        background: linear-gradient(
            135deg,
            #3B82F6,
            #2563EB
        ) !important;
    }}


    /* =====================================================
       دکمه مشاهده فیلم
    ===================================================== */

    .watch-button button {{
        background: #2563EB !important;

        color: white !important;

        border: 1px solid #2563EB !important;

        border-radius: 9px !important;

        font-weight: bold !important;

        min-height: 42px !important;
    }}

    .watch-button button:hover {{
        background: #1D4ED8 !important;

        border-color: #1D4ED8 !important;
    }}


    /* =====================================================
       دکمه بازگشت
    ===================================================== */

    .back-button button {{
        border-radius: 9px !important;

        font-weight: bold !important;
    }}


    /* =====================================================
       پیام‌ها
    ===================================================== */

    [data-testid="stAlert"] {{
        direction: rtl !important;

        text-align: right !important;
    }}


    /* =====================================================
       فوتر
    ===================================================== */

    .footer {{
        text-align: center !important;

        direction: rtl !important;

        color: #64748B;

        margin-top: 45px;

        padding-top: 20px;

        border-top: 1px solid {border_color};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# دکمه شب و روز - بالا سمت راست
# =========================================================

theme_col, empty_col = st.columns(
    [1, 11]
)

with theme_col:

    st.markdown(
        '<div class="theme-button">',
        unsafe_allow_html=True
    )

    if st.session_state.dark_mode:

        theme_clicked = st.button(
            "☀️ روز",
            key="theme_button",
            use_container_width=True
        )

    else:

        theme_clicked = st.button(
            "🌙 شب",
            key="theme_button",
            use_container_width=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


if theme_clicked:

    # فقط ظاهر برنامه را تغییر می‌دهیم.
    # هیچ‌کدام از وضعیت‌های جستجو، پیشنهادها یا صفحه فعلی پاک نمی‌شوند.
    st.session_state.dark_mode = (
        not st.session_state.dark_mode
    )

    # Streamlit بعد از کلیک خودش اسکریپت را دوباره اجرا می‌کند؛
    # بنابراین نیازی به st.rerun() نداریم.

# =========================================================
# ذخیره صفحه فعلی در History
# =========================================================

def save_current_page():

    st.session_state.page_history.append({

        "selected_movie_index":
            st.session_state.selected_movie_index,

        "recommendations":
            st.session_state.recommendations,

        "selected_search_movie":
            st.session_state.selected_search_movie,

        "movie_search":
            st.session_state.get(
                "movie_search",
                ""
            ),

        "movie_selectbox":
            st.session_state.get(
                "movie_selectbox",
                None
            ),

        "recommendation_slider":
            st.session_state.get(
                "recommendation_slider",
                10
            )
    })

# =========================================================
# بازگشت به صفحه قبلی
# =========================================================

def go_back():

    if not st.session_state.page_history:
        return

    previous_page = (st.session_state.page_history.pop())

    st.session_state.selected_movie_index = (previous_page["selected_movie_index"])

    st.session_state.recommendations = (previous_page["recommendations"])

    st.session_state.selected_search_movie = (previous_page["selected_search_movie"])

    st.session_state.movie_search = (previous_page["movie_search"])

    if previous_page["movie_selectbox"] is not None:

        st.session_state.movie_selectbox = (previous_page["movie_selectbox"])

    st.session_state.recommendation_slider = (previous_page["recommendation_slider"])

    st.session_state.current_page = "home"
    st.rerun()
# =========================================================
# بارگذاری دیتاست
# =========================================================

@st.cache_data
def load_movies():

    start = time.time()

    movies = pd.read_parquet(
        "data/movies.parquet"
    )

    end = time.time()

    print(
        f"⏱ زمان بارگذاری دیتاست: "
        f"{end - start:.2f} ثانیه"
    )

    return movies


# =========================================================
# بارگذاری TF-IDF
# =========================================================

@st.cache_resource
def load_model():

    start = time.time()

    print(
        "شروع بارگذاری tfidf_matrix..."
    )

    data = np.load(
        "tfidf_data.npy",
        mmap_mode="r"
    )

    indices = np.load(
        "tfidf_indices.npy",
        mmap_mode="r"
    )

    indptr = np.load(
        "tfidf_indptr.npy",
        mmap_mode="r"
    )

    with open(
        "tfidf_shape.txt",
        "r"
    ) as f:

        shape = tuple(
            map(
                int,
                f.read().split(",")
            )
        )

    tfidf_matrix = csr_matrix(
        (
            data,
            indices,
            indptr
        ),
        shape=shape
    )

    end = time.time()

    print(
        f"tfidf_matrix بارگذاری شد: "
        f"{end - start:.2f} ثانیه"
    )

    return tfidf_matrix


# =========================================================
# اجرای بارگذاری
# =========================================================

movies = load_movies()


tfidf_matrix = load_model()


# =========================================================
# عنوان اصلی برنامه
# =========================================================

st.markdown(
    """
    <div class="main-title">
         سیستم پیشنهاد فیلم🎬
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
        جستجو کنید و فیلم‌های مشابه را کشف کنید
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# صفحه جزئیات یک فیلم
# =========================================================

if st.session_state.current_page == "movie":
    selected_index = (
        st.session_state.selected_movie_index
    )

    movie = movies.iloc[
        selected_index
    ]


    # -----------------------------------------------------
    # دکمه بازگشت
    # -----------------------------------------------------

    st.markdown(
        '<div class="back-button">',
        unsafe_allow_html=True
    )

    if st.button(
        "⬅️ بازگشت",
        key="back_button"
    ):
        st.session_state.current_page = "home"
        st.session_state.selected_movie_index = None
        st.rerun()



    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # کارت جزئیات
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-movie-card">',
        unsafe_allow_html=True
    )

    # RTL:
    # ستون اول = اطلاعات
    # ستون دوم = پوستر
    info_col, poster_col = st.columns(
        [3, 1]
    )


    # -----------------------------------------------------
    # پوستر - سمت راست
    # -----------------------------------------------------

    with poster_col:

        if pd.notna(
            movie.get("poster_path")
        ):

            st.image(
                "https://image.tmdb.org/t/p/w500"
                + str(
                    movie["poster_path"]
                ),
                use_container_width=True
            )

        else:

            st.info(
                "پوستر این فیلم موجود نیست."
            )


    # -----------------------------------------------------
    # اطلاعات - سمت چپ
    # -----------------------------------------------------

    with info_col:

        st.markdown(
            f"""
            <div class="main-movie-title">
                {movie['title']}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <span class="info-box">
                ⭐ امتیاز:
                {movie['vote_average']}
            </span>

            <span class="info-box">
                🗳 تعداد رأی:
                {movie['vote_count']:,}
            </span>

            <span class="info-box">
                📅 تاریخ انتشار:
                {movie['release_date']}
            </span>
            """,
            unsafe_allow_html=True
        )


        if pd.notna(
            movie.get("genres")
        ):

            st.markdown(
                f"""
                <span class="info-box">
                    🎭 ژانر:
                    {movie['genres']}
                </span>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            """
            <div class="main-movie-title"
                 style="font-size:22px;margin-top:20px;">
                📖 توضیحات فیلم
            </div>
            """,
            unsafe_allow_html=True
        )


        if pd.notna(
            movie.get("overview")
        ):

            st.markdown(
                f"""
                <div class="movie-description">
                    {movie['overview']}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="movie-description">
                    توضیحاتی برای این فیلم وجود ندارد.
                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# صفحه اصلی جستجو
# =========================================================

else:

    # -----------------------------------------------------
    # Search Box
    # -----------------------------------------------------

    search = st.text_input(
        "عنوان فیلم را جستجو کنید",
        key="movie_search"
    )

    # اگر به دلیل rerun مقدار کادر جستجو موقتاً خالی شد،
    # آخرین فیلمی که برای آن پیشنهاد ساخته شده را برمی‌گردانیم.
    if (
        not search
        and st.session_state.selected_search_movie is not None
        and st.session_state.recommendations is not None
    ):
        search = st.session_state.selected_search_movie


    # -----------------------------------------------------
    # جستجو
    # -----------------------------------------------------

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


        # -------------------------------------------------
        # فیلم پیدا نشد
        # -------------------------------------------------

        if results.empty:

            st.warning(
                "فیلمی با این عنوان پیدا نشد."
            )


        # -------------------------------------------------
        # فیلم پیدا شد
        # -------------------------------------------------

        else:

            st.markdown(
                f"""
                <div style="
                    text-align:right;
                    direction:rtl;
                    color:{secondary_text};
                    margin-bottom:15px;
                    font-size:15px;
                ">
                    {len(results):,} فیلم پیدا شد.
                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------------------------------
            # انتخاب فیلم + تعداد پیشنهاد
            # -------------------------------------------------

            select_col, slider_col, button_col = st.columns(
                [3, 2, 2]
            )


            # -------------------------------------------------
            # انتخاب فیلم
            # -------------------------------------------------

            with select_col:

                movie_titles = results["title"].tolist()

                    # اگر فیلم قبلی دیگر در نتایج وجود نداشت
                if (
                        st.session_state.selected_movie_title
                        not in movie_titles
                ):
                        st.session_state.selected_movie_title = movie_titles[0]


                selected_movie = st.selectbox(
                        "🎬 فیلم موردنظر خود را انتخاب کنید",
                        movie_titles,
                        key="selected_movie_title"
                    )


            # -------------------------------------------------
            # تعداد پیشنهاد
            # -------------------------------------------------


            with slider_col:

                number = st.slider(
                    "تعداد فیلم های مشابه",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="recommendation_number"
                )


            # -------------------------------------------------
            # دکمه پیشنهاد
            # -------------------------------------------------

            with button_col:

                st.markdown(
                    '<div class="recommend-button">',
                    unsafe_allow_html=True
                )

                recommend_button = st.button(
                    "🎯 پیشنهاد فیلم‌های مشابه",
                    key="recommend_button",
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # پیدا کردن index
            # -------------------------------------------------

            movie_index = results[
                results["title"]
                == selected_movie
            ].index[0]


            movie = movies.loc[
                movie_index
            ]


            # -------------------------------------------------
            # کارت فیلم انتخاب شده
            # -------------------------------------------------

            st.markdown(
                '<div class="main-movie-card">',
                unsafe_allow_html=True
            )


            # اطلاعات چپ + پوستر راست
            info_col, poster_col = st.columns(
                [3, 1]
            )


            # -------------------------------------------------
            # پوستر
            # -------------------------------------------------

            with poster_col:

                if pd.notna(
                    movie.get("poster_path")
                ):

                    st.image(
                        "https://image.tmdb.org/t/p/w500"
                        + str(
                            movie["poster_path"]
                        ),
                        use_container_width=True
                    )

                else:

                    st.info(
                        "پوستر این فیلم موجود نیست."
                    )


            # -------------------------------------------------
            # اطلاعات فیلم
            # -------------------------------------------------

            with info_col:

                st.markdown(
                    f"""
                    <div class="main-movie-title">
                        {movie['title']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
                    <span class="info-box">
                        ⭐ امتیاز:
                        {movie['vote_average']}
                    </span>

                    <span class="info-box">
                        🗳 تعداد رأی:
                        {movie['vote_count']:,}
                    </span>

                    <span class="info-box">
                        📅 تاریخ انتشار:
                        {movie['release_date']}
                    </span>
                    """,
                    unsafe_allow_html=True
                )


                if pd.notna(
                    movie.get("genres")
                ):

                    st.markdown(
                        f"""
                        <span class="info-box">
                            🎭 ژانر:
                            {movie['genres']}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )


                st.markdown(
                    """
                    <div class="main-movie-title"
                         style="
                            font-size:22px;
                            margin-top:20px;
                         ">
                        📖 توضیحات فیلم
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                if pd.notna(
                    movie.get("overview")
                ):

                    st.markdown(
                        f"""
                        <div class="movie-description">
                            {movie['overview']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="movie-description">
                            توضیحاتی برای این فیلم وجود ندارد.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =================================================
            # محاسبه پیشنهادها
            # =================================================

            if recommend_button:

                with st.spinner(
                    "در حال پیدا کردن فیلم‌های مشابه..."
                ):

                    start = time.time()


                    similarities = sp_matmul_topn(
                        tfidf_matrix[
                            movie_index
                        ],
                        tfidf_matrix.T,
                        top_n=number + 1,
                        threshold=0.0,
                        sort=True,
                        n_threads=-1
                    )


                    end = time.time()


                    print(
                        f"⏱ زمان پیدا کردن فیلم‌های مشابه: "
                        f"{end - start:.2f} ثانیه"
                    )


                    recommendations = []


                    # -------------------------------------------------
                    # ساخت لیست پیشنهادها
                    # -------------------------------------------------

                    for index, similarity in zip(
                        similarities.indices,
                        similarities.data
                    ):

                        index = int(
                            index
                        )


                        if index == movie_index:

                            continue


                        recommendations.append({

                            "index":
                                index,

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
                                    float(
                                        similarity
                                    ) * 100,
                                    2
                                )

                        })


                        if len(
                            recommendations
                        ) >= number:

                            break


                # -------------------------------------------------
                # ذخیره
                # -------------------------------------------------

                st.session_state.recommendations = (
                    recommendations
                )

                st.session_state.selected_search_movie = (
                    selected_movie
                )
                st.session_state.current_page = "home"

                st.rerun()


            # =================================================
            # نمایش پیشنهادها
            # =================================================

            if (
                st.session_state.recommendations
                is not None
            ):

                st.markdown(
                    """
                    <div class="recommendation-section-title">
                        🎯 فیلم‌های پیشنهادی مشابه
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                recommendations = (
                    st.session_state.recommendations
                )


                # -------------------------------------------------
                # هر ردیف = 5 فیلم
                # -------------------------------------------------

                for row_start in range(
                    0,
                    len(recommendations),
                    5
                ):

                    row = recommendations[
                        row_start:row_start + 5
                    ]


                    columns = st.columns(
                        5
                    )


                    for position, (
                        column,
                        recommendation
                    ) in enumerate(
                        zip(
                            columns,
                            row
                        )
                    ):

                        recommendation_index = int(
                            recommendation["index"]
                        )


                        recommendation_movie = (
                            movies.iloc[
                                recommendation_index
                            ]
                        )


                        with column:

                            st.markdown(
                                '<div class="recommendation-card">',
                                unsafe_allow_html=True
                            )


                            # -------------------------------------------------
                            # پوستر
                            # -------------------------------------------------

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
                                    ),
                                    use_container_width=True
                                )

                            else:

                                st.markdown(
                                    """
                                    <div style="
                                        height:300px;
                                        display:flex;
                                        align-items:center;
                                        justify-content:center;
                                        border:1px solid #334155;
                                        border-radius:10px;
                                        color:#94A3B8;
                                    ">
                                        پوستر موجود نیست
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )


                            # -------------------------------------------------
                            # عنوان
                            # -------------------------------------------------

                            st.markdown(
                                f"""
                                <div class="recommendation-title">
                                    {recommendation_movie['title']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            # -------------------------------------------------
                            # تاریخ
                            # -------------------------------------------------

                            st.markdown(
                                f"""
                                <div class="recommendation-info">
                                    📅 تاریخ انتشار:
                                    {recommendation_movie['release_date']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            # -------------------------------------------------
                            # امتیاز
                            # -------------------------------------------------

                            st.markdown(
                                f"""
                                <div class="recommendation-info">
                                    ⭐ امتیاز:
                                    {recommendation_movie['vote_average']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            # -------------------------------------------------
                            # شباهت
                            # -------------------------------------------------

                            st.markdown(
                                f"""
                                <div class="similarity">
                                    🎯 شباهت:
                                    {recommendation['similarity']}%
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            # -------------------------------------------------
                            # دکمه مشاهده
                            # -------------------------------------------------

                            st.markdown(
                                '<div class="watch-button">',
                                unsafe_allow_html=True
                            )


                            if st.button(
                                "مشاهده فیلم",
                                key=(
                                    f"watch_"
                                    f"{recommendation_index}_"
                                    f"{row_start}_"
                                    f"{position}"
                                ),
                                use_container_width=True
                            ):

                                st.session_state.selected_movie_index = (
                                    recommendation_index
                                )

                                st.session_state.current_page = "movie"

                                st.rerun()


                            st.markdown(
                                '</div>',
                                unsafe_allow_html=True
                            )


                            st.markdown(
                                '</div>',
                                unsafe_allow_html=True
                            )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        ساخته شده با ❤️ توسط
        <span style="color:#3B82F6;font-weight:bold;">
            Moein Ghafari
        </span>
    </div>
    """,
    unsafe_allow_html=True
)