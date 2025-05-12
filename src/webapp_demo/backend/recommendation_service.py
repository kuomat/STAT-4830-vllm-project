# backend/recommendation_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from recommendation import content_filtering_recommend, collaborative_filtering_recommend, low_rank_recommend, two_tower_recommend
import csv
import os

class Req(BaseModel):
    selected: list[int]    # note: we use ints here
    n: int

app = FastAPI()
BASE = os.path.dirname(__file__)
METADATA_CSV = os.path.join(BASE, "..", "..", "..", "dataset", "embeddings_final.csv")

@app.get("/items")
async def get_all_items():
    items = []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "id":          int(row["image_key"]),
                "name":        row.get("name", ""),
                "description": row.get("description", ""),
                "price":       float(row.get("price") or 0),
            })
    return items

@app.post("/recommend/{method}")
async def recommend(method: str, req: Req):
    try:
        user_map = {k: +1 for k in req.selected}

        if method == 'content-filtering':
            recs = content_filtering_recommend(user_map, req.n)
        elif method == 'collaborative-filtering':
            recs = collaborative_filtering_recommend(user_map, req.n)
        elif method == 'low-rank':
            recs = low_rank_recommend(user_map, req.n)
        elif method == 'two-tower':
            recs = two_tower_recommend(user_map, req.n)

        else:
            return {"error": f"unsupported method: {method}"}, 400

        return {"recommendations": recs}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Recommendation service error: {str(e)}"}, 500