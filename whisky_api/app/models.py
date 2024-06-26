from typing import Optional

from sqlmodel import Field, SQLModel

# --------aroma models--------
class Le_nez_du_whisky_aromaBase(SQLModel):
    aroma_name: str
    aroma_notes: str
    aroma_annotation: str
    aroma_description: str
    aroma_file_location: str
    aroma_tic_location: str

class Le_nez_du_whisky_aroma(Le_nez_du_whisky_aromaBase, table=True):
    aroma_id: Optional[int] = Field(default=None, primary_key=True)
    
class Le_nez_du_whisky_aromaCreate(Le_nez_du_whisky_aromaBase):
    aroma_id: int
    
class Le_nez_du_whisky_aromaRead(Le_nez_du_whisky_aromaBase):
    aroma_id: int

# --------whisky models--------
class WhiskyBase(SQLModel):
    whisky_brand: Optional[str] = Field(default=None)
    whisky_name: str = Field(default=None)
    whisky_age: Optional[int] = Field(default=None)
    whisky_alcohol_perc: Optional[int] = Field(default=None)
    whisky_distillation_freq: Optional[str] = Field(default=None)
    whisky_distillation_date: Optional[str] = Field(default=None)
    whisky_bottle_date: Optional[str] = Field(default=None)
    whisky_filtration: Optional[str] = Field(default=None)
    whisky_grain: Optional[str] = Field(default=None)
    whisky_barrel: Optional[str] = Field(default=None)
    whisky_distillation_technique: Optional[str] = Field(default=None)
    whisky_img_loc: Optional[str] = Field(default=None)
    
class Whisky(WhiskyBase, table=True):
    whisky_code: Optional[str] = Field(default=None, primary_key=True)

class WhiskyCreate(WhiskyBase):
    whisky_code: Optional[str] = Field(default=None)

class WhiskyRead(WhiskyBase):
    whisky_code: str
    
# --------sample models--------    
class SampleBase(SQLModel):
    sample_code: str = Field(index=True)
    sample_file_location: str
    sample_tic_location: str
    

class Sample(SampleBase, table=True):
    sample_id: Optional[int] = Field(default=None, primary_key=True)
    
    
class SampleCreate(SampleBase):
    pass
    
    
class SampleRead(SampleBase):
    sample_id: int
    
# --------peak models--------    
class PeakBase(SQLModel):
    peak_sample_id: Optional[int] = Field(default=None, foreign_key="sample.sample_id")
    peak_aroma_id: Optional[int] = Field(default=None, foreign_key="le_nez_du_whisky_aroma.aroma_id")
    peak_splash: str = Field(index=True)
    peak_retention_time: float
    peak_area: int
    peak_masses: str
    peak_intensities: str
    peak_height: int
    peak_compound: str

    
class Peak(PeakBase, table=True):
    peak_id: Optional[int] = Field(default=None, primary_key=True)
    
class PeakCreate(PeakBase):
    pass
    
class PeakRead(PeakBase):
    peak_id: Optional[int] = Field(default=None)
    
    

    
