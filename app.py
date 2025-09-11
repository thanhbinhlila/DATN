
import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, unicodedata

DATA_DIR = Path("./data/")
p_info = DATA_DIR / "hotel_info.csv"
p_comments = DATA_DIR / "hotel_comments.csv"

OUTPUT_DIR = Path("./output/")
MODELS_DIR = OUTPUT_DIR / "models"

def read_csv_safely(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1258", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

hotel_info = read_csv_safely(p_info)
hotel_comments = read_csv_safely(p_comments)

text_cols = [c for c in hotel_info.columns if re.search(r"(Description)", c, re.I)]
id_cols   = [c for c in hotel_info.columns if re.search(r"(Hotel_ID)", c, re.I)]
name_cols = [c for c in hotel_info.columns if re.search(r"(Hotel_Name)", c, re.I)]
HOTEL_ID_COL   = id_cols[0] if id_cols else hotel_info.columns[0]
HOTEL_NAME_COL = name_cols[0] if name_cols else (hotel_info.columns[1] if hotel_info.shape[1] > 1 else HOTEL_ID_COL)

def simple_clean(s):
    if not isinstance(s, str): s = str(s)
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"http\\S+|www\\S+", " ", s)
    s = re.sub(r"[\\w.-]+@[\\w.-]+", " ", s)
    s = re.sub(r"\\d+", " ", s)
    s = re.sub(r"[^\\w\\sáàảãạăằắẳẵặâầấẩẫậéèẻẽẹêềếểễệíìỉĩịóòỏõọôồốổỗộơờớởỡợúùủũụưừứửữựýỳỷỹỵđ]", " ", s, flags=re.I)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

hotel_info["_text_info"] = (hotel_info[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else "")
corpus = hotel_info["_text_info"].map(simple_clean).fillna("")
tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1,2), min_df=2)
X = tfidf.fit_transform(corpus)
COS = cosine_similarity(X)

st.set_page_config(page_title="Agoda RS Mini", layout="wide")
st.title("Agoda Recommendation System")

method = st.sidebar.selectbox("Methods", ["TF-IDF", "Doc2Vec", "SBERT", "ALS"])

seed = st.selectbox("Choose Base Hotel:", hotel_info[HOTEL_NAME_COL].astype(str).tolist())
seed_id = hotel_info.loc[hotel_info[HOTEL_NAME_COL]==seed, HOTEL_ID_COL].iloc[0]
idx = hotel_info.index[hotel_info[HOTEL_ID_COL]==seed_id][0]

def show_recs(df):
    st.write(df.reset_index(drop=True))

if method == "TF-IDF":
    sims = COS[idx]
    top_idx = np.argsort(-sims)[:11]
    out = hotel_info.iloc[top_idx][[HOTEL_ID_COL, HOTEL_NAME_COL]].copy()
    out["similarity"] = sims[top_idx]
    out = out[out.index != idx].head(10)
    show_recs(out)

elif method == "Doc2Vec":
    try:
        D2V_EMB = np.load(MODELS_DIR / "d2v_emb.npy")
        sims = cosine_similarity([D2V_EMB[idx]], D2V_EMB)[0]
        top_idx = np.argsort(-sims)[:11]
        out = hotel_info.iloc[top_idx][[HOTEL_ID_COL, HOTEL_NAME_COL]].copy()
        out["similarity"] = sims[top_idx]
        out = out[out.index != idx].head(10)
        show_recs(out)
    except Exception as e:
        st.warning("Doc2Vec embedding's not ready yet!!!")

elif method == "SBERT":
    try:
        SBERT_EMB = np.load(MODELS_DIR / "sbert_emb.npy")
        sims = cosine_similarity([SBERT_EMB[idx]], SBERT_EMB)[0]
        top_idx = np.argsort(-sims)[:11]
        out = hotel_info.iloc[top_idx][[HOTEL_ID_COL, HOTEL_NAME_COL]].copy()
        out["similarity"] = sims[top_idx]
        out = out[out.index != idx].head(10)
        show_recs(out)
    except Exception as e:
        st.warning("SBERT embedding's not ready yet!!!.")

elif method == "ALS":
    st.write("Input User ID for Suggestion:")
    user_id = st.number_input("User ID:", min_value=0, step=1, value=0)
    try:
        import findspark
        findspark.init()

        from pyspark import SparkContext
        from pyspark.ml.recommendation import ALSModel
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import *

        SparkContext.setSystemProperty('spark.hadoop.dfs.client.use.datanode.hostname', 'true')
        sc = SparkContext(master="local", appName="New Spark Context")
        sc.setLogLevel("ERROR")


        spark = SparkSession(sc)
        spark

        model_path = MODELS_DIR / "best_als_model"
        model = ALSModel.load(str(model_path))
        user_df = spark.createDataFrame([(user_id,)], ["userId"])
        recs = model.recommendForUserSubset(user_df, 10).toPandas()
        if recs.empty:
            st.info("No suggestion for this User ID.")
        else:
            rows = []
            for arr in recs["recommendations"].iloc[0]:
                rows.append({"itemId": arr["itemId"], "score": float(arr["rating"])})
            out = pd.DataFrame(rows)
            try:
                out = out.merge(hotel_info[[HOTEL_ID_COL, HOTEL_NAME_COL]].astype({HOTEL_ID_COL: out["itemId"].dtype}), 
                                left_on="itemId", right_on=HOTEL_ID_COL, how="left")
            except:
                out = out.merge(hotel_info[[HOTEL_ID_COL, HOTEL_NAME_COL]], 
                                left_on="itemId", right_on=HOTEL_ID_COL, how="left")
            show_recs(out)
            spark.stop()
            sc.stop()
    except Exception as e:
        spark.stop()
        sc.stop()
        st.warning("ALS model's not ready yet!!!.")
