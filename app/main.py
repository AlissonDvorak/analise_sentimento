from fastapi import FastAPI,  Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from routers.input_audio import input_audio
from routers.input_texto import input_texto 
import uvicorn


app = FastAPI()

@app.get("/", tags=["Página Inicial"])
def main():
    return Response(status_code=200)




app.include_router(input_texto)
app.include_router(input_audio)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)