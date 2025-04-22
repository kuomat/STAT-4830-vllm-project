# backend/recommendation_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from recommendation import content_filtering_recommend, collaborative_filtering_recommend, low_rank_recommend, two_tower_recommend

class Req(BaseModel):
    selected: list[int]    # note: we use ints here
    n: int

app = FastAPI()

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