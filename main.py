from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import os
import aiohttp
import json
from pathlib import Path
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create index if it doesn't exist
if 'resume-analysis' not in pc.list_indexes().names():
    pc.create_index(
        name='resume-analysis',
        dimension=768,  # dimension for all-mpnet-base-v2 model
        metric='cosine'
    )

# Initialize embeddings model
embeddings_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

class ResumeUploadResponse(BaseModel):
    resume_id: str
    filename: str
    status: str

class AnalysisRequest(BaseModel):
    resume_ids: List[str]
    job_requirements: str

class ResumeAnalysis(BaseModel):
    resume_id: str
    filename: str
    match_score: float
    qualifications_match: List[str]
    missing_requirements: List[str]
    additional_skills: List[str]
    years_experience: float
    summary: str

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_with_deepseek(resume_content: str, job_requirements: str) -> dict:
    """Analyze resume using DeepSeek model via OpenRouter with retry logic"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    api_base = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Resume Analysis App"
    }
    
    # Optimized prompt for reliability
    shortened_prompt = f"""
Analyze resume for frontend role. Return JSON only.

Requirements:
{job_requirements[:300]}

Resume:
{resume_content[:500]}

JSON format:
{{
    "match_score": <0-100>,
    "qualifications_match": ["qual1", "qual2"],
    "missing_requirements": ["req1", "req2"],
    "additional_skills": ["skill1", "skill2"],
    "years_experience": <number>,
    "summary": "<brief summary>"
}}
"""
    
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": shortened_prompt}],
        "temperature": 0.2,
        "max_tokens": 300,
        "response_format": { "type": "json_object" }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_base, headers=headers, json=payload) as response:
                result = await response.json()
                
                # Debug logging
                print(f"API Response: {result}")
                
                try:
                    content = result['choices'][0]['message']['content']
                    
                    # Handle empty response
                    if not content.strip():
                        raise ValueError("Empty response from API")
                    
                    # Remove any markdown formatting
                    if content.startswith('```json'):
                        content = content[7:-3]
                    elif content.startswith('```'):
                        content = content[3:-3]
                    
                    # Clean the JSON string
                    content = content.strip()
                    
                    # Parse JSON with error handling
                    try:
                        analysis = json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}")
                        # Try to fix common JSON issues
                        if content.endswith(','):
                            content = content[:-1]
                        if not content.endswith('}'):
                            content += '}'
                        analysis = json.loads(content)
                    
                    # Validate and ensure required fields
                    required_fields = {
                        'match_score': 0,
                        'qualifications_match': [],
                        'missing_requirements': [],
                        'additional_skills': [],
                        'years_experience': 0,
                        'summary': ''
                    }
                    
                    # Merge with defaults for missing fields
                    analysis = {**required_fields, **analysis}
                    
                    # Validate types
                    analysis['match_score'] = float(analysis['match_score'])
                    analysis['years_experience'] = float(analysis['years_experience'])
                    analysis['qualifications_match'] = list(analysis['qualifications_match'])
                    analysis['missing_requirements'] = list(analysis['missing_requirements'])
                    analysis['additional_skills'] = list(analysis['additional_skills'])
                    analysis['summary'] = str(analysis['summary'])
                    
                    return analysis
                    
                except Exception as e:
                    print(f"Error processing API response: {e}")
                    print(f"Raw content: {content if 'content' in locals() else 'No content'}")
                    raise  # Allow retry
                    
    except Exception as e:
        print(f"API call error: {e}")
        raise  # Allow retry

async def store_resume_vectors(resume_content: str, resume_id: str, filename: str):
    """Store resume vectors in Pinecone"""
    embeddings = embeddings_model.encode(resume_content)
    
    index = pc.Index('resume-analysis')
    
    try:
        index.upsert(
            vectors=[(
                resume_id, 
                embeddings.tolist(), 
                {
                    "content": resume_content,
                    "filename": filename
                }
            )]
        )
    except Exception as e:
        print(f"Error storing vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing resume vectors: {str(e)}")

async def get_resume_content(resume_id: str) -> Dict:
    """Retrieve resume content from Pinecone"""
    index = pc.Index('resume-analysis')
    query_response = index.fetch(ids=[resume_id])
    
    vectors = query_response.vectors
    
    if resume_id not in vectors:
        raise HTTPException(status_code=404, detail=f"Resume with ID {resume_id} not found")
        
    return {
        "content": vectors[resume_id].metadata['content'],
        "filename": vectors[resume_id].metadata['filename']
    }

@app.post("/upload", response_model=List[ResumeUploadResponse])
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload resumes and store their vectors"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
        
    responses = []
    
    for file in files:
        try:
            # Generate unique ID for resume
            resume_id = f"resume_{os.urandom(8).hex()}"
            
            # Read and decode file content
            content = await file.read()
            try:
                resume_content = content.decode('utf-8')
            except UnicodeDecodeError:
                resume_content = content.decode('latin-1')
            
            # Save file to uploads directory
            file_path = UPLOAD_DIR / f"{resume_id}_{file.filename}"
            with open(file_path, "wb") as f:
                await file.seek(0)
                file_content = await file.read()
                f.write(file_content)
            
            # Store vectors
            await store_resume_vectors(resume_content, resume_id, file.filename)
            
            responses.append(ResumeUploadResponse(
                resume_id=resume_id,
                filename=file.filename,
                status="uploaded"
            ))
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            responses.append(ResumeUploadResponse(
                resume_id="error",
                filename=file.filename,
                status=f"error: {str(e)}"
            ))
    
    return responses

@app.post("/analyze", response_model=Dict)
async def analyze_resumes(request: AnalysisRequest):
    """Analyze previously uploaded resumes against job requirements"""
    if not request.resume_ids:
        raise HTTPException(status_code=400, detail="No resume IDs provided")
        
    results = []
    
    for resume_id in request.resume_ids:
        try:
            # Retrieve resume content
            resume_data = await get_resume_content(resume_id)
            resume_content = resume_data["content"]
            filename = resume_data["filename"]
            
            # Analyze with DeepSeek (with retries)
            analysis = await analyze_with_deepseek(resume_content, request.job_requirements)
            
            # Add resume identifier
            analysis["resume_id"] = resume_id
            analysis["filename"] = filename
            results.append(analysis)
            
        except HTTPException as e:
            results.append({
                "resume_id": resume_id,
                "error": str(e.detail)
            })
        except Exception as e:
            print(f"Error analyzing resume {resume_id}: {e}")
            results.append({
                "resume_id": resume_id,
                "error": f"Analysis error: {str(e)}"
            })
    
    # Sort results by match score
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x['match_score'], reverse=True)
    
    return {
        "analysis": sorted_results,
        "total_resumes": len(valid_results),
        "errors": [r for r in results if "error" in r]
    }

@app.get("/resumes")
async def list_resumes():
    """List all uploaded resumes"""
    try:
        index = pc.Index('resume-analysis')
        stats = index.describe_index_stats()
        query_response = index.fetch(ids=[str(i) for i in range(stats.total_vector_count)])
        
        return [{
            "resume_id": id,
            "filename": vector.metadata['filename']
        } for id, vector in query_response.vectors.items()]
    except Exception as e:
        print(f"Error listing resumes: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing resumes: {str(e)}")

@app.get("/status")
async def check_status():
    """Check system status"""
    try:
        pc.list_indexes()
        pinecone_status = True
    except Exception as e:
        print(f"Pinecone connection error: {e}")
        pinecone_status = False
    
    api_key_status = bool(os.getenv('OPENROUTER_API_KEY'))
    upload_dir_status = UPLOAD_DIR.exists()
    
    return {
        "initialized": True,
        "api_status": {
            "pinecone": pinecone_status,
            "openrouter": api_key_status
        },
        "upload_directory": {
            "exists": upload_dir_status,
            "path": str(UPLOAD_DIR.absolute())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)