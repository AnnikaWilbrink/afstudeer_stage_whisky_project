from fastapi import FastAPI, Depends, Query
from sqlmodel import Session, select
from typing import Optional, List

from .models import Le_nez_du_whisky_aroma, Le_nez_du_whisky_aromaRead, Le_nez_du_whisky_aromaCreate, Sample, SampleRead, SampleCreate, Peak, PeakRead, PeakCreate, Whisky, WhiskyRead, WhiskyCreate
from .database import engine, create_db_and_tables #, database
from .dependencies import get_session

app = FastAPI()

session = Session(bind=engine)

# CREATE DB AND TABLES ON START UP     
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# CREATE PEAK
@app.post('/peaks/', response_model=PeakRead)
def create_peak(*, session: Session = Depends(get_session), pc: PeakCreate):
    db_peak = Peak.from_orm(pc)
    session.add(db_peak)
    session.commit()
    session.refresh(db_peak)
    return db_peak

# GET ONE PEAK FROM A SAMPLE OR AN AROMA FROM DATABASE
@app.get('/peaks/', response_model=List[PeakRead])
def read_peaks(*, session: Session = Depends(get_session), sample_id: int = Query(default=None), 
               aroma_id: int = Query(default=None)):
    statement = select(Peak)
    if sample_id:
        statement = statement.filter(Peak.peak_sample_id == sample_id)
    if aroma_id:
        statement = statement.filter(Peak.peak_aroma_id == aroma_id)
    return session.exec(statement).all()

# GET ALL PEAKS FROM A SAMPLE OR AN AROMA FROM DATABASE
@app.get('/total_peaks/', response_model=List[PeakRead])
def read_total_peaks(*, session: Session = Depends(get_session), aroma_id: int = Query(default=None),
                     sample_id: int = Query(default=None)):
    statement = select(Peak)
    if aroma_id:
        statement = statement.filter(Peak.peak_aroma_id == aroma_id).order_by(Peak.peak_retention_time)
    if sample_id:
        statement = statement.filter(Peak.peak_sample_id == sample_id).order_by(Peak.peak_retention_time)
    return session.exec(statement).all()

# CREATE SAMPLE
@app.post('/samples/', response_model=SampleRead)
def create_sample(*, session: Session = Depends(get_session), sc: SampleCreate):
    db_sample = Sample.from_orm(sc)
    session.add(db_sample)
    session.commit()
    session.refresh(db_sample)
    return db_sample

# GET ONE SAMPLE FROM DATABASE WITH SAMPLE_CODE
@app.get('/samples/{code}', response_model=List[SampleRead])
def read_samples(*, session: Session = Depends(get_session), code: str):
    sample = session.query(Sample).filter(Sample.sample_code == code).all()
    return sample

# GET ONE SAMPLE FROM DATABASE
@app.get('/sample/', response_model=List[SampleRead])
def read_sample(*, session: Session = Depends(get_session), sample_id: int = Query(default=None)):
    statement = select(Sample).filter(Sample.sample_id == sample_id)
    return session.exec(statement).all()

# CREATE AROMA
@app.post('/aromas/', response_model=Le_nez_du_whisky_aromaRead)
def create_aroma(*, session: Session = Depends(get_session), ac: Le_nez_du_whisky_aromaCreate):
    db_aroma = Le_nez_du_whisky_aroma.from_orm(ac)
    session.add(db_aroma)
    session.commit()
    session.refresh(db_aroma)
    return db_aroma

# GET ALL AROMAS FROM DATABASE
@app.get('/aromas/', response_model=List[Le_nez_du_whisky_aromaRead])
def read_aromas(*, session: Session = Depends(get_session)):
    statement = select(Le_nez_du_whisky_aroma)
    aromas =  session.exec(statement).all()
    return aromas

# GET ONE AROMA FROM DATABASE
@app.get('/aroma/', response_model=List[Le_nez_du_whisky_aromaRead])
def read_aroma(*, session: Session = Depends(get_session), aroma_id: int = Query(default=None)):
    statement = select(Le_nez_du_whisky_aroma).filter(Le_nez_du_whisky_aroma.aroma_id == aroma_id)
    return session.exec(statement).all()

# ??
@app.delete("/aromas/{aroma_id}")
def delete_aroma(aroma_id: int):
    aroma = session.get(Le_nez_du_whisky_aroma, aroma_id)
    session.delete(aroma)
    session.commit()
    return {"ok": True}

 # CREATE WHISKY       
@app.post('/whisky/', response_model=WhiskyRead)
def create_whisky(*, session: Session = Depends(get_session), wc: WhiskyCreate):
    db_whisky = Whisky.from_orm(wc)
    session.add(db_whisky)
    session.commit()
    session.refresh(db_whisky)
    return db_whisky

# GET ALL WHISKIES FROM DATABASE
@app.get('/whiskys/', response_model=List[WhiskyRead])
def read_whiskys(*, session: Session = Depends(get_session)):
    statement = select(Whisky)
    whiskys =  session.exec(statement).all()
    return whiskys

# GET ONE WHISKY FROM DATABASE
@app.get('/whisky/', response_model=List[WhiskyRead])
def read_whisky(*, session: Session = Depends(get_session), whisky_code: str = Query(default=None)):
    statement = select(Whisky).filter(Whisky.whisky_code == whisky_code)
    return session.exec(statement).all()






