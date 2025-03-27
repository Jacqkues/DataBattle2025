from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from openai import OpenAI
from qdrant_client import QdrantClient
from services.genai import text_to_text, text_image_to_text , analyse_answer , generate_question , rewrite_question
from services.retrivial import Retrivial
import gc
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import os
from utils import encode_img, load_json
from fastapi.templating import Jinja2Templates
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text,select, func
from fastapi.staticfiles import StaticFiles
import random
import markdown
from fastembed import LateInteractionTextEmbedding

NUM_PAGE_TO_FETCH = 2


category = ["Filing Requirements & Formalities", "Priority Claims & Right of Priority", "Divisional Applications", "Fees, Payment & Time Limits", "Languages & Translations","Procedural Remedies & Legal Effect","PCT Procedure & Entry into EP Phase","Examination, Amendments, and Grant","Opposition & Appeals","Substantive Patent Law","Entitlement & Transfers"]
base_image_path = "/app/data/pdf-images"
base_url = "/data/pdf-images"
question_file = "/app/data/question-categories/categorie.json"


Base = declarative_base()
class Evaluator(Base):
    __tablename__ = "evaluators"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    closed_questions = Column(Boolean, default=False)
    scoring = Column(String)
    criteria = Column(String)

class PageLink(Base):
    __tablename__ = "pagelink"
    id = Column(Integer, primary_key=True, index=True)
    pdf_path = Column(Text)
    evaluator_id = Column(Integer)

class Question(Base):
    __tablename__ = "question"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    category = Column(Text)

class QuestionLink(Base):
    __tablename__ = "questionlink"
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer)
    evaluator_id = Column(Integer)

class QuestionPageLink(Base):
    __tablename__ = "question_page_link"
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer)
    page = Column(Text)







SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)



templates = Jinja2Templates(directory="templates")





@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print("load open router client")
    app.open_router_api = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=os.getenv("OPENROUTER_API"))
    print("load groq client")
    app.groq_api = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API"))

    print("connecting to qdrant")
    app.qdrant_client = QdrantClient(
        url=os.getenv("DATABASE_URL"),
        api_key=os.getenv("DATABASE_KEY"),
    )

    print("Init retrivial model ...")

    colqwen_model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.1")

   

    print("Retrivial model ok !")

    print("Question retrivial init...")
    colbert = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    app.retrivial = Retrivial(colbert,processor,colqwen_model,app.qdrant_client)

    print("Init sql base ")
    app.db = SessionLocal()

    print("Check if question is empty")
    if not app.db.query(Question).first():
        print("init question...")
        data = load_json(question_file)
        for item in data["content"]:
            theme = item["theme"].split(". ", 1)[-1]

            print(f"------{theme}-------------")
            for question in item["questions"]:
                    q = Question(question=question,category=theme)
                    app.db.add(q)
                    app.db.commit()
                    app.db.refresh(q)
                    print("retrieve pages ...")
                    pages = app.retrivial.fetch_doc_page(question,None,NUM_PAGE_TO_FETCH)[0]
                    print(f"fetch {len(pages.points)} pages ")
                    for page in pages.points:
                        link = QuestionPageLink(question_id = q.id, page=os.path.join(base_image_path, page.payload["pdf"], page.payload["page"]))
                        app.db.add(link)
                        app.db.commit()


    yield

    colqwen_model = None
    processor = None
    colbert = None

    app.db.close()
    app = None
    gc.collect()


app = FastAPI(lifespan=lifespan)

app.mount("/data", StaticFiles(directory="data"), name="data")


@app.get("/evaluator")
async def root(request: Request,response_class=HTMLResponse):
    evaluators = app.db.query(Evaluator).order_by(Evaluator.id.desc()).all()
    return templates.TemplateResponse("index.html", {"request": request, "evaluators": evaluators})

@app.get("/", response_class=HTMLResponse)
async def chatbot(request: Request):

    return templates.TemplateResponse("chatbot.html", {"request": request})




@app.get("/create_evaluator", response_class=HTMLResponse)
async def new_evaluator(request: Request):


    return templates.TemplateResponse("create_evaluator.html", {"request": request, "message": "Create Evaluator","categorys":category})


@app.post("/create_evaluator")
async def create_evaluator(
    name: str = Form(...),
    description:str = Form(...),
    closed_questions:str = Form("false"),
    scoring:str = Form(...),
    criteria:str = Form(...),
):
    print("creating evaluator...")
    evaluator = Evaluator(
        name=name,
        description=description,
        closed_questions=(closed_questions.lower() == "true"),
        scoring=scoring,
        criteria=criteria
    )
    app.db.add(evaluator)
    app.db.commit()
    app.db.refresh(evaluator)

    print("fetching related question ...")

    questions = app.db.query(Question).filter(Question.category == description).order_by(func.random()).limit(2).all()

    print(f"----------------Fetched {questions}---------------------------------")

    for question in questions:
        app.db.add(QuestionLink(question_id=question.id,evaluator_id = evaluator.id))
        app.db.commit()
        app.db.refresh(question)
        print("fetching related page...")

        #pages = app.retrivial.fetch_doc_page(question.question,None,NUM_PAGE_TO_FETCH)[0]
        #pages = app.db.query(QuestionPageLink).filter(QuestionLink.question_id == question.id).all()
        pages = (
            app.db.query(QuestionPageLink)
            .join(QuestionLink, QuestionPageLink.question_id == QuestionLink.question_id)
            .filter(QuestionLink.question_id == question.id)
            .all()
        )
        for page in pages:
            #change by 
            link = PageLink(
                pdf_path=page.page,
                evaluator_id=evaluator.id
            )

            app.db.add(link)
            app.db.commit()
            app.db.refresh(link)

            print(f"{link.evaluator_id} - {link.pdf_path}")

    return RedirectResponse(url="/evaluator", status_code=303)

@app.get("/edit_evaluator/{evaluator_id}", response_class=HTMLResponse)
async def edit_evaluator(evaluator_id: int, request: Request):
    evaluator = app.db.query(Evaluator).filter(Evaluator.id == evaluator_id).first()
    stmt = select(PageLink).where(PageLink.evaluator_id == evaluator_id)
    items = app.db.scalars(stmt).all()

    pdf_path = {}
    for item in items:
        source = item.pdf_path.split('/')

        page = source[-1]
        file = source[-2]
        formatted_string = file + " " + page.replace('.png',"").replace('_',' ')
        url = os.path.join(base_url,file,page)
        pdf_path[formatted_string] = url
        #pdf_path.append(item.pdf_path)

    return templates.TemplateResponse("edit_evaluator.html", {"request": request, "evaluator": evaluator,"sources":pdf_path})

@app.post("/delete_evaluator/{evaluator_id}")
async def delete_evaluator(evaluator_id: int):
    evaluator = app.db.query(Evaluator).filter(Evaluator.id == evaluator_id).first()
    if evaluator:
        app.db.delete(evaluator)
        app.db.query(PageLink).filter(PageLink.evaluator_id == evaluator_id).delete()
        app.db.commit()
    return RedirectResponse(url="/evaluator", status_code=303)

@app.get("/evaluator/{evaluator_id}", response_class=HTMLResponse)
async def read_evaluator(request: Request, evaluator_id: int):
    evaluator = app.db.query(Evaluator).filter(Evaluator.id == evaluator_id).first()
    return templates.TemplateResponse("chat.html", {"request": request, "evaluator": evaluator})

@app.get("/examen_question/{evaluator_id}")
async def get_quest(request:Request,evaluator_id:int):
    evaluator = app.db.query(Evaluator).filter(Evaluator.id == evaluator_id).first()


    question_ids = app.db.scalars(
        select(QuestionLink.question_id).where(QuestionLink.evaluator_id == evaluator_id)
    ).all()

    # Return None if no questions are linked
    if not question_ids:
        return None

    random_question_id = random.choice(question_ids)

    question = app.db.scalar(select(Question).where(Question.id == random_question_id))

    #todo check which is better


    data = app.retrivial.fetch_question(question.question)

    #data = app.retrivial.fetch_question(evaluator.description)
    data = random.choice(data.points).payload
    return templates.TemplateResponse("question.html", {"evaluator": evaluator_id, "request": request, "question":data.get("question") , "image_url":"test/test/test","answer":data.get('answer'),"legal_basis":"notdefineds"})



@app.get('/generate/{evaluator_id}',response_class=HTMLResponse)
async def generate(request:Request,evaluator_id:int):

    stmt = select(PageLink).where(PageLink.evaluator_id == evaluator_id)
    
    items = app.db.scalars(stmt).all()
    evaluator = app.db.query(Evaluator).filter(Evaluator.id == evaluator_id).first()



    print(items)


    select_item = random.choice(items)
    #select_item = items[0]
    base64_img = encode_img(select_item.pdf_path)


    data = generate_question(app.open_router_api,base64_img,evaluator.description,evaluator.closed_questions)


    return templates.TemplateResponse("question.html", {"evaluator": evaluator_id, "request": request, "question":data.get("question") , "image_url":select_item.pdf_path,"answer":data.get('answer'),"legal_basis":data.get('legal_basis')})



@app.get("/search_pages")
async def search_pages(query:str):
    return {"pages": app.retrivial.fetch_doc_page(query,None)}

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(query: str = Form(...)):
    #rewrited_q = rewrite_question(app.groq_api,query)
    #print(rewrited_q)
    

    first_page = app.retrivial.fetch_doc_page(query,None)[0].points[0].payload

    image_path  = os.path.join(base_image_path, first_page["pdf"], first_page["page"])


    base64_img = encode_img(image_path)

    output = text_image_to_text(app.open_router_api,query,base64_img,"google/gemma-3-4b-it:free")

    output = markdown.markdown(output)

    source = image_path.split('/')

    page = source[-1]
    file = source[-2]
    formatted_string = file + " " + page.replace('.png',"").replace('_',' ')
    url = os.path.join(base_url,file,page)
    source_link = f"<a class='underline' href='{url}'>{formatted_string}</a>"

    #response = f"<p><strong>Answer:</strong> You asked: {output}</p>"
    response = f"<div class='h-[300px] overflow-y-auto prose'>{output}</div><p class='mt-4 font-bold'>Source</p>{source_link}"
    return response




@app.post("/analyse_answer/{evaluator_id}",response_class=HTMLResponse)
async def analyse_answer_route(
    request:Request,
    evaluator_id:int,
    question:str = Form(...),
    real_answer:str = Form(...),
    legal_doc:str = Form(...),
    user_answer:str = Form(...),
    image_url:str = Form(...),
    

):
    evaluator = app.db.query(Evaluator).filter(Evaluator.id == evaluator_id).first()
    result = analyse_answer(app.groq_api,question,user_answer,real_answer + " justification : " + legal_doc,evaluator.criteria,evaluator.scoring)

    print(result)

    html_content = markdown.markdown(result.get('feedback'))
    
    source = image_url.split('/')

    page = source[-1]
    file = source[-2]
    formatted_string = file + " " + page.replace('.png',"").replace('_',' ')

    url = os.path.join(base_url,file,page)
    return templates.TemplateResponse("feedback.html", {"request": request,"evaluator":evaluator_id, "question":question ,"feedback":html_content,"score":result.get('score'),"user_answer":user_answer,"image_name":formatted_string,"image_url":url})
    

