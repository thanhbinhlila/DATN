# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import os
import glob
from pathlib import Path

# [OPTIMIZED] Import các thư viện ML/DL ở đây
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# ---------------- Cấu hình trang (Page Config) ----------------
st.set_page_config(
    page_title="Agoda Hotel Recommendation",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

import random # [NEW] Thêm thư viện random

# ... các dòng import khác ...

# [NEW] Tạo một danh sách chứa các link ảnh mặc định
FALLBACK_IMAGE_URLS = [
    "https://images.pexels.com/photos/1268871/pexels-photo-1268871.jpeg",
    "https://images.pexels.com/photos/1179156/pexels-photo-1179156.jpeg",
    "https://images.pexels.com/photos/33818491/pexels-photo-33818491.jpeg",
    "https://images.pexels.com/photos/4825701/pexels-photo-4825701.jpeg",
    "https://images.pexels.com/photos/2684260/pexels-photo-2684260.jpeg",
    "https://images.pexels.com/photos/2992314/pexels-photo-2992314.jpeg",
    "https://images.pexels.com/photos/940856/pexels-photo-940856.jpeg",
    "https://images.pexels.com/photos/59924/pexels-photo-59924.jpeg",
    "https://images.pexels.com/photos/2983472/pexels-photo-2983472.jpeg",
    "https://images.pexels.com/photos/2506988/pexels-photo-2506988.jpeg"
]

# ---------------- [OPTIMIZED] Hằng số và đường dẫn ----------------
DATA_PATH = Path("./data")
MODEL_PATH = Path("./outputs/models")
HOTEL_INFO_FILE = DATA_PATH / "hotel_info.csv"
HOTEL_COMMENTS_FILE = DATA_PATH / "hotel_comments.csv"
# ALS_MODEL_PATH = MODEL_PATH / "best_als_model"
ALS_RECOMMENDATIONS_FILE = DATA_PATH / "als_recommendations.csv"
D2V_EMBEDDINGS_FILE = MODEL_PATH / "d2v_emb.npy"
SBERT_EMBEDDINGS_FILE = MODEL_PATH / "sbert_emb.npy"


# ---------------- [OPTIMIZED] Quản lý Spark Session ----------------
def get_spark_session():
    """
    Khởi tạo và trả về một SparkSession, cache trong st.session_state.
    """
    if "spark" not in st.session_state:
        st.session_state.spark = (
            SparkSession.builder
            .master("local[*]")
            .appName("AgodaALSApp")
            .config("spark.driver.memory", "4g")
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
            .config("spark.hadoop.fs.file.impl.disable.cache", "true")
            .getOrCreate()
        )
        # Ép lại cấu hình trong HadoopConf để đảm bảo
        st.session_state.spark.sparkContext._jsc.hadoopConfiguration().set(
            "fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem"
        )
    return st.session_state.spark

def _delete_crc_files(dir_path: str):
    """Xóa các file .crc checksum để tránh lỗi khi Spark đọc từ local."""
    for fp in glob.glob(os.path.join(dir_path, "**", "*.crc"), recursive=True):
        try:
            os.remove(fp)
        except OSError as e:
            st.warning(f"Không thể xóa file CRC {fp}: {e}")


# ---------------- Các hàm tải và xử lý dữ liệu (giữ nguyên logic) ----------------
# Các hàm gốc được giữ nguyên, chỉ thêm cache vào session_state
def normalize_col(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')

def auto_rename_columns(df: pd.DataFrame, wanted: dict) -> pd.DataFrame:
    norm_map = {normalize_col(c): c for c in df.columns}
    rename_dict = {}
    for target, cands in wanted.items():
        cands_norm = [normalize_col(x) for x in cands + [target]]
        for k in cands_norm:
            if k in norm_map:
                rename_dict[norm_map[k]] = target
                break
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df

def load_main_data():
    try:
        hotel_df = pd.read_csv(HOTEL_INFO_FILE, encoding="utf-8")
    except Exception:
        hotel_df = pd.read_csv(HOTEL_INFO_FILE, encoding="latin-1")
    hotel_df.columns = hotel_df.columns.str.strip()
    hotel_df = auto_rename_columns(
        hotel_df,
        {
            "Hotel_ID": ["hotel_id", "hotelid"], "Hotel_Name": ["hotel_name", "name"],
            "Hotel_Address": ["hotel_address", "address"], "Hotel_Description": ["hotel_description", "description"],
            "Image_URL": ["image_url", "image"], "Hotel_Rank": ["hotel_rank", "rank", "stars"],
            "Total_Score": ["total_score", "score"]
        }
    )
    if "Hotel_Description" not in hotel_df.columns:
        cols = [c for c in ["Hotel_Name", "Hotel_Address"] if c in hotel_df.columns]
        hotel_df["Hotel_Description"] = hotel_df[cols].astype(str).fillna("").agg(" ".join, axis=1) if cols else ""
    if "Total_Score" not in hotel_df.columns:
        hotel_df["Total_Score"] = np.random.uniform(7.5, 9.8, size=len(hotel_df)).round(1)

    try:
        comments_df = pd.read_csv(HOTEL_COMMENTS_FILE, encoding="utf-8")
    except Exception:
        comments_df = pd.read_csv(HOTEL_COMMENTS_FILE, encoding="latin-1")
    comments_df.columns = comments_df.columns.str.strip()
    comments_df = auto_rename_columns(comments_df, {"Reviewer_Name": ["reviewer_name", "user_name"], "Hotel_ID": ["hotel_id"]})
    
    # [FIX] Thêm 2 dòng sau để làm sạch cột Reviewer_Name
    comments_df.dropna(subset=['Reviewer_Name'], inplace=True)
    comments_df['Reviewer_Name'] = comments_df['Reviewer_Name'].astype(str)

    missing = [c for c in ["Reviewer_Name", "Hotel_ID"] if c not in comments_df.columns]
    if missing:
        st.error(f"Thiếu cột bắt buộc trong file `{HOTEL_COMMENTS_FILE.name}`: {', '.join(missing)}")
        st.stop()
        
    return hotel_df, comments_df

def create_tfidf_recommender(hotel_df):
    corpus = hotel_df["Hotel_Description"].astype(str).fillna("").apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def load_embeddings(embedding_file):
    try:
        return np.load(embedding_file)
    except FileNotFoundError:
        st.warning(f"Không tìm thấy file embedding tại: {embedding_file}")
        return None

def preprocess_text(text):
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_content_recommendations(hotel_index, sim_matrix, df, top_n=10):
    sim_scores = sorted(list(enumerate(sim_matrix[hotel_index])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    hotel_indices = [i[0] for i in sim_scores]
    return df.iloc[hotel_indices]


# ---------------- [OPTIMIZED] Hàm khởi tạo state ----------------
def initialize_session_state():
    """
    Tải tất cả dữ liệu và model vào st.session_state để tránh tải lại.
    """
    if "initialized" in st.session_state:
        return

    with st.spinner("Đang chuẩn bị dữ liệu..."):
        st.session_state.hotel_df, st.session_state.comments_df = load_main_data()
        
        # Content-based models (giữ nguyên)
        tfidf_recommender = create_tfidf_recommender(st.session_state.hotel_df)
        st.session_state.tfidf_vectorizer = tfidf_recommender[0]
        st.session_state.tfidf_matrix = tfidf_recommender[1]
        st.session_state.tfidf_cosine_sim = tfidf_recommender[2]
        
        st.session_state.d2v_embeddings = load_embeddings(D2V_EMBEDDINGS_FILE)
        st.session_state.sbert_embeddings = load_embeddings(SBERT_EMBEDDINGS_FILE)
        
        # [REPLACED] Tải file gợi ý ALS đã được tính toán sẵn
        try:
            st.session_state.als_recs_df = pd.read_csv(ALS_RECOMMENDATIONS_FILE)
        except FileNotFoundError:
            st.error(f"Lỗi: Không tìm thấy file `{ALS_RECOMMENDATIONS_FILE}`.")
            # Tạo DataFrame rỗng để tránh lỗi ở các phần khác
            st.session_state.als_recs_df = pd.DataFrame(columns=['UserName', 'RecommendedHotel'])

        # Tạo sẵn danh sách hotel và user để dùng trong UI
        st.session_state.hotel_names = st.session_state.hotel_df['Hotel_Name'].unique()
        st.session_state.user_list = sorted(st.session_state.comments_df['Reviewer_Name'].unique())

    st.session_state.initialized = True
    st.success("Sẵn sàng!", icon="✅")


# ---------------- [OPTIMIZED] Giao diện hiển thị gợi ý ----------------
# [FIX] Thay thế toàn bộ hàm cũ bằng hàm này
def display_recommendation_list(df_recommendations):
    """
    Hiển thị danh sách khách sạn được gợi ý với giao diện chuyên nghiệp hơn.
    """
    if df_recommendations.empty:
        st.info("Không tìm thấy gợi ý nào phù hợp.")
        return

    num_recommendations = len(df_recommendations)
    num_rows = (num_recommendations + 2) // 3
    recommendations_iterator = df_recommendations.iterrows()

    for i in range(num_rows):
        cols = st.columns(3)
        for j in range(3):
            try:
                index, data = next(recommendations_iterator)
                with cols[j]:
                    with st.container(border=True):
                        fallback_image = random.choice(FALLBACK_IMAGE_URLS)
                        image_url = data.get('Image_URL', fallback_image)
                        
                        st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                        st.image(image_url, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.subheader(data['Hotel_Name'])
                        st.caption(f"📍 {data.get('Hotel_Address', 'N/A')}")

                        metric_cols = st.columns(2)

                        with metric_cols[0]:
                            rank_value = str(data.get('Hotel_Rank', 'N/A')).strip()
                            st.markdown("⭐ **Hạng**") # Dùng label chung để thẳng hàng
                            if 'no info' in rank_value.lower() or 'n/a' in rank_value.lower():
                                st.markdown(f"<p style='font-size:12px !important; color: #808080;'><i>Không có</i></p>", unsafe_allow_html=True)
                            else:
                                # [MODIFIED] Thay thế st.metric bằng st.markdown với font-size tùy chỉnh
                                st.markdown(f"<p style='font-size: 16px; font-weight: bold;'>{rank_value}</p>", unsafe_allow_html=True)
                        
                        with metric_cols[1]:
                            score_value = str(data.get('Total_Score', 'N/A')).strip()
                            st.markdown("💯 **Điểm**") # Dùng label chung để thẳng hàng
                            if 'no info' in score_value.lower() or 'n/a' in score_value.lower():
                                st.markdown(f"<p style='font-size:12px !important; color: #808080;'><i>Không có</i></p>", unsafe_allow_html=True)
                            else:
                                # [MODIFIED] Thay thế st.metric bằng st.markdown với font-size tùy chỉnh
                                st.markdown(f"<p style='font-size: 16px; font-weight: bold;'>{score_value}</p>", unsafe_allow_html=True)

                        with st.expander("Xem mô tả"):
                            st.write(data.get('Hotel_Description', 'Không có mô tả.'))
            except StopIteration:
                break

# ---------------- [OPTIMIZED] Các hàm render cho từng trang ----------------
def render_page_by_description():
    st.header("🔎 Tìm kiếm theo mô tả khách sạn")
    st.write("Nhập những tiện ích hoặc đặc điểm bạn mong muốn, hệ thống sẽ tìm các khách sạn phù hợp nhất.")
    
    search_query = st.text_input(
        "Ví dụ: khách sạn có hồ bơi cho gia đình, gần trung tâm, yên tĩnh",
        placeholder="Nhập mô tả của bạn ở đây...",
        label_visibility="collapsed"
    )
    
    if search_query:
        with st.spinner("Đang tìm những khách sạn phù hợp nhất..."):
            query_vec = st.session_state.tfidf_vectorizer.transform([preprocess_text(search_query)])
            sim_scores = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
            top_indices = sim_scores.argsort()[-9:][::-1] # Lấy 9 gợi ý
            recommendations = st.session_state.hotel_df.iloc[top_indices]
        
        st.markdown("---")
        st.subheader("🏆 Kết quả gợi ý hàng đầu")
        display_recommendation_list(recommendations)

def render_page_by_name():
    st.header("🏨 Tìm khách sạn tương tự")
    st.write("Chọn một khách sạn bạn đã từng thích, hệ thống sẽ gợi ý những nơi có đặc điểm tương đồng.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_hotel_name = st.selectbox(
            "Chọn một khách sạn bạn đã thích:",
            st.session_state.hotel_names,
            index=None,
            placeholder="Tìm kiếm khách sạn..."
        )
    with col2:
        method = st.selectbox(
            "Chọn mô hình gợi ý:",
            ["TF-IDF", "Doc2Vec", "SBERT"]
        )

    if selected_hotel_name:
        selected_hotel_index = st.session_state.hotel_df[st.session_state.hotel_df['Hotel_Name'] == selected_hotel_name].index[0]
        
        with st.spinner(f"Đang tìm các khách sạn tương tự bằng {method}..."):
            recommendations = pd.DataFrame()
            sim_matrix = None

            if method == "TF-IDF":
                sim_matrix = st.session_state.tfidf_cosine_sim
            elif method == "Doc2Vec" and st.session_state.d2v_embeddings is not None:
                sim_matrix = cosine_similarity(st.session_state.d2v_embeddings)
            elif method == "SBERT" and st.session_state.sbert_embeddings is not None:
                sim_matrix = cosine_similarity(st.session_state.sbert_embeddings)
            
            if sim_matrix is not None:
                recommendations = get_content_recommendations(selected_hotel_index, sim_matrix, st.session_state.hotel_df, top_n=9)
        
        st.markdown("---")
        st.subheader(f"Top 9 gợi ý tương tự '{selected_hotel_name}'")
        display_recommendation_list(recommendations)

def render_page_by_als():
    st.header("👤 Gợi ý cá nhân hóa (từ File có sẵn)")
    st.info("Tính năng này đọc kết quả gợi ý đã được tính toán trước từ file `als_recommendations.csv`.")

    # Lấy danh sách user CÓ trong file gợi ý để lựa chọn
    available_users = sorted(st.session_state.als_recs_df['UserName'].unique())
    
    # search_user = st.text_input("Tìm kiếm tên khách hàng:", placeholder="Nhập tên để tìm...")
    # if search_user:
    #     filtered_users = [user for user in available_users if search_user.lower() in user.lower()]
    # else:
    filtered_users = available_users

    selected_user = st.selectbox("Chọn một khách hàng để xem gợi ý:", filtered_users, index=None, placeholder="Chọn một khách hàng...")

    if selected_user and st.button(f"🚀 Lấy gợi ý cho {selected_user}", type="primary", use_container_width=True):
        with st.spinner("Đang lấy dữ liệu gợi ý..."):
            # [REPLACED] Lọc DataFrame thay vì chạy model
            recs_df = st.session_state.als_recs_df[st.session_state.als_recs_df['UserName'] == selected_user]

            if not recs_df.empty:
                # Trộn kết quả với thông tin khách sạn đầy đủ
                merged_df = recs_df.merge(
                    st.session_state.hotel_df,
                    left_on='RecommendedHotel',
                    right_on='Hotel_Name',
                    how='inner'  # Chỉ giữ lại những khách sạn có trong cả 2 file
                )
                
                st.markdown("---")
                st.subheader(f"✨ Gợi ý dành riêng cho '{selected_user}'")
                display_recommendation_list(merged_df)
            else:
                st.warning(f"Không tìm thấy gợi ý nào cho khách hàng '{selected_user}' trong file.")

# ---------------- Chương trình chính ----------------
def main():
    st.title("🏨 AGODA Hotel Recommendation System")
    st.caption("Ứng dụng gợi ý khách sạn sử dụng các mô hình lọc nội dung và lọc cộng tác.")
    
    # [NEW] Thêm thông tin GVHD và HV thực hiện
    st.markdown("<p style='color: #007bff; font-weight: bold;'>GVHD: Khuất Thùy Phương</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #007bff; font-weight: bold;'>HV thực hiện: Nguyễn Thanh Bình - Nguyễn Tuấn Duy</p>", unsafe_allow_html=True)

    # [NEW] Thêm CSS để chỉnh kích thước ảnh đồng đều
    st.markdown("""
        <style>
            .image-container img {
                width: 100%;
                height: 200px; /* <- Bạn có thể thay đổi chiều cao này */
                object-fit: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    # Khởi tạo dữ liệu và model một lần duy nhất
    initialize_session_state()

    with st.sidebar:
        st.image("logo.png", width=100)
        st.header("Phương thức gợi ý")
        
        page_options = {
            "Theo mô tả khách sạn": render_page_by_description,
            "Theo khách sạn tương tự": render_page_by_name,
            "Theo đánh giá người dùng (ALS)": render_page_by_als,
        }
        
        selected_page = st.radio(
            "Chọn trang:",
            options=page_options.keys(),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.header("Về dự án")
        st.info("Đây là đồ án tốt nghiệp ứng dụng các thuật toán gợi ý vào bài toán thực tế trên dữ liệu từ Agoda.")

    # Gọi hàm render tương ứng với trang đã chọn
    page_options[selected_page]()

if __name__ == "__main__":
    main()


















