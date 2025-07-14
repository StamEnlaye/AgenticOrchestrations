from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from query_preprocessing.outlinesTesting import generate_decomposition, StructuredOutput
from query_preprocessing.fullAgentImplementation import decompose, plan_subquery2, missingInfo


app = FastAPI()


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class PromptRequest(BaseModel):
    prompt: str


class SubqueryResponse(BaseModel):
    subqueries: list[str]

class MissingInfoResponse(BaseModel):
    sufficient: bool
    missingInfo: List[str] | None = None

class GenerationRequest(BaseModel):
    query: str
    generatedResponse: str

# POST endpoint for `decompose`
@app.post("/juliette")
def decomp(req: PromptRequest):
    output = decompose("qwen3:4b", req.prompt)
    return output

# POST endpoint for `generate_decomposition`
@app.post("/outlinesDecomp", response_model=StructuredOutput)
def decomp2(req: PromptRequest) -> StructuredOutput:
    output = generate_decomposition("qwen3:4b", req.prompt)
    return output


@app.post("/subqueryDirect", response_model=SubqueryResponse)
def subquery_direct(req: PromptRequest):
    return plan_subquery2("qwen3:4b", req.prompt)


@app.post("/missingInfo", response_model=MissingInfoResponse)
def missing_info_endpoint(req: GenerationRequest) -> MissingInfoResponse:
    result = missingInfo("qwen3:4b", req.query, req.generatedResponse)
    print(f"Missing info result: {result}")
    return result