# backend/recommendation_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from recommendation import content_filtering_recommend

class Req(BaseModel):
    selected: list[int]    # note: we use ints here
    n: int

app = FastAPI()

@app.post("/recommend/{method}")
async def recommend(method: str, req: Req):
    if method != 'content-filtering':
        return {"error": "only 'content-filtering' implemented"}, 400

    # build dict of image_key -> +1/-1 from the list of selected keys
    # here you might encode selection=+1, unselected=-1 as per your logic
    user_map = {k: +1 for k in req.selected}
    recs = content_filtering_recommend(user_map, req.n)
    return {"recommendations": recs}
