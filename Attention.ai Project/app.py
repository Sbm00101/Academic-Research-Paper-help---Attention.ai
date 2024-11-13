from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qa_pipeline import answer_question  # Import your Q&A function from the Q&A agent

# Initialize FastAPI app
app = FastAPI()


# Define request model for the Q&A endpoint
class QueryRequest(BaseModel):
    query: str


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Academic Research Assistant API!"}


# Q&A endpoint
@app.post("/answer")
async def get_answer(request: QueryRequest):
    query = request.query
    try:
        # Use the Q&A function to get an answer, including references
        result = answer_question(query)
        answer = result.get("answer")
        references = result.get("references")

        return {
            "query": query,
            "answer": answer,
            "references": references
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
