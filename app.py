import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger

CONN = os.environ['CONN']

MODEL_PATH = os.environ['MODEL_PATH']

app = FastAPI()

engine = create_engine(CONN, pool_size=5, max_overflow=-1)


def batch_load_sql(query: str):
    chunksize = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=chunksize):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():
    logger.info('loading liked posts')
    liked_posts = batch_load_sql("""SELECT distinct post_id, user_id FROM public.feed_data WHERE action='like' """)

    logger.info('loading posts features')
    posts_features = pd.read_sql(
        """SELECT * FROM "n-ignatov-12.2" """,

        con=CONN
    )

    logger.info('loading user features')
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con=CONN
    )

    return [liked_posts, posts_features, user_features]


def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH, format='cbm')

    return model


logger.info('loading model')
model = load_model()
logger.info('loading features')
features = load_features()
logger.info('service is up and running')


def get_recommended_feed(id: int, time: datetime, limit: int):
    logger.info(f'user_id: {id}')
    logger.info('reading features')
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    logger.info('dropping columns')
    posts_features = features[1].drop('text', axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    logger.info('zipping data')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))

    logger.info('assign')
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    logger.info('predicting')
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    logger.info('deleting liked posts')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_post(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)
