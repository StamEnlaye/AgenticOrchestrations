from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from outlinesTesting import generate_decomposition, StructuredOutput
from fullAgentImplementation import decompose

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


#change get to post, change string in the body like get the prompt in the request body