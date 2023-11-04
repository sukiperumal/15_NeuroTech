from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from drowsiness_detection import detect_drowsiness  # Your drowsiness detection function

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.post("/detect_drowsiness/")
async def detect_drowsiness_post(request: Request, username: str = Form(...)):
    result = detect_drowsiness()  # Call your drowsiness detection function
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
