import hashlib
import json
import logging
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from backend.app.config import ROOT, CONFIG
from backend.app.errors import AppError
from backend.app.models import ComponentsUpdate
from backend.app import store
from backend.app.profiles import confirm_components, contour_proposals
from backend.app.pipeline import analyze, annotate
from backend.app.reports import pdf_report
from backend.schematic.parser import parse_parts, placement_proposals
from backend.vision.alignment import decode_image, rectify

logging.basicConfig(level=logging.INFO,format='%(message)s')

@asynccontextmanager
async def lifespan(app):
    store.init()
    app.state.executor = ThreadPoolExecutor(max_workers=CONFIG['runtime']['workers'])
    yield
    app.state.executor.shutdown(wait=True)

app = FastAPI(title='PCB Missing-Component Detector',version='1.0.0',lifespan=lifespan)

@app.exception_handler(AppError)
async def app_error(request, exc):
    return JSONResponse(status_code=exc.status,content={'error':{'code':exc.code,'message':exc.message}})

@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse(status_code=422,content={'error':{'code':'validation_error','message':'Check the form fields or component boxes.',
        'details':[{'location':list(e['loc']),'message':e['msg']} for e in exc.errors()]}})

@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logging.getLogger('pcb.api').exception(json.dumps(dict(event='request_failed',path=request.url.path)))
    return JSONResponse(status_code=500,content={'error':{'code':'internal_error','message':'The request could not complete. Check the server log and retry.'}})

@app.middleware('http')
async def limit_request_size(request:Request,call_next):
    length = request.headers.get('content-length')
    if length and (not length.isdigit() or int(length)>CONFIG['images']['max_upload_mb']*1024*1024*4):
        return JSONResponse(status_code=413,content={'error':{'code':'request_too_large','message':'Request is too large. Resize the images or split uploads.'}})
    return await call_next(request)

async def read_upload(upload: UploadFile, extensions):
    filename = upload.filename or ''
    extension = Path(filename).suffix.lower()
    if extension not in extensions:
        raise AppError('unsupported_file',f"Unsupported file type. Allowed: {', '.join(sorted(extensions))}")
    limit = CONFIG['images']['max_upload_mb']*1024*1024
    raw = await upload.read(limit+1)
    if not raw: raise AppError('empty_file','Uploaded file is empty.')
    if len(raw)>limit: raise AppError('file_too_large',f'Each file must be at most {CONFIG["images"]["max_upload_mb"]} MB.',413)
    return raw

IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.heic','.heif'}
SCHEMATIC_EXTENSIONS = {'.kicad_sch','.net','.csv'}
REFERENCE_EXTENSIONS = {'.pdf','.jpg','.jpeg','.png','.heic','.heif'}

def create_profile(name,raw,parts,width_mm,height_mm,placement=None):
    if not all(math.isfinite(v) and v>0 for v in [width_mm,height_mm]):
        raise AppError('invalid_holder','Holder dimensions must be positive finite numbers.')
    if len(parts)>CONFIG['detection']['max_components']:
        raise AppError('too_many_components','This board exceeds the configured component limit.')
    image = decode_image(raw)
    rectified,homography,centers = rectify(image,width_mm,height_mm)
    if placement:
        parts = placement_proposals(*placement,parts,(rectified.shape[1],rectified.shape[0]))
    board_id = uuid.uuid4().hex
    profile = dict(id=board_id,name=name[:128],width_mm=width_mm,height_mm=height_mm,components=parts,
                   confirmed=False,ideal_hash=hashlib.sha256(raw).hexdigest(),
                   canonical_size=[rectified.shape[1],rectified.shape[0]],proposals=contour_proposals(rectified),
                   warnings=['Confirm every box on the rectified ideal image before analysis.'])
    directory = store.DATA/'boards'/board_id
    directory.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(directory/'ideal.png'),rectified)
    cv2.imwrite(str(directory/'original.png'),image)
    (directory/'profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    store.save_board(profile)
    return profile

@app.get('/api/health')
def health():
    return {'status':'ok','features':CONFIG['features']}

@app.get('/api/boards')
def list_boards():
    return [{'id':b['id'],'name':b['name'],'confirmed':b['confirmed'],'count':len(b['components'])} for b in store.boards()]

@app.post('/api/boards',status_code=201)
async def new_board(ideal:Annotated[UploadFile,File()],
    schematic:Annotated[UploadFile | None,File()]=None,bom:Annotated[UploadFile | None,File()]=None,
    placement:Annotated[UploadFile | None,File()]=None,
    name:Annotated[str,Form(min_length=1,max_length=128)]='Untitled board',
    width_mm:Annotated[float,Form(gt=0)]=CONFIG['geometry']['baseline_width_mm'],
    height_mm:Annotated[float,Form(gt=0)]=CONFIG['geometry']['baseline_height_mm']):
    raw = await read_upload(ideal,IMAGE_EXTENSIONS)
    if not schematic and not bom: raise AppError('schematic_required','Upload a KiCad schematic/netlist or BOM CSV.')
    parts, source_files, reference_only = [], [], False
    if schematic:
        content = await read_upload(schematic,SCHEMATIC_EXTENSIONS|REFERENCE_EXTENSIONS)
        source_files.append(('schematic'+Path(schematic.filename).suffix.lower(),content))
        if Path(schematic.filename).suffix.lower() in REFERENCE_EXTENSIONS:
            reference_only = True
            if not bom: raise AppError('bom_required','PDF/image schematics cannot supply a netlist. Attach a BOM CSV and define component boxes.')
        else: parts = parse_parts(schematic.filename,content)
    if bom:
        content = await read_upload(bom,{'.csv'})
        bom_parts = parse_parts(bom.filename,content)
        source_files.append(('bom.csv',content))
        if not parts: parts = bom_parts
        else:
            extra = {p['designator']:p for p in bom_parts}
            for part in parts:
                supplement = extra.get(part['designator'],{})
                for key in ['value','footprint','mpn']:
                    if supplement.get(key): part[key] = supplement[key]
    placement_data = None
    if placement:
        content = await read_upload(placement,{'.kicad_pcb','.csv'})
        placement_data = (placement.filename,content)
        source_files.append(('placement'+Path(placement.filename).suffix.lower(),content))
    # CPU work is off the event loop.
    import asyncio
    profile = await asyncio.to_thread(create_profile,name,raw,parts,width_mm,height_mm,placement_data)
    for filename,content in source_files: (store.DATA/'boards'/profile['id']/filename).write_bytes(content)
    if reference_only: profile['warnings'].append('Schematic attachment is reference-only; connectivity comes from a netlist, not a PDF/image.')
    if schematic and schematic.filename.lower().endswith('.kicad_sch'):
        profile['warnings'].append('Symbol properties imported. Export a KiCad XML .net file for connectivity and flattened hierarchical/multi-unit parts.')
    store.save_board(profile)
    return profile

@app.get('/api/boards/{board_id}')
def get_board(board_id:str):
    return store.board(board_id)

@app.get('/api/boards/{board_id}/image')
def board_image(board_id:str):
    board = store.board(board_id)
    return FileResponse(store.DATA/'boards'/board['id']/'ideal.png',media_type='image/png')

@app.put('/api/boards/{board_id}/components')
def save_components(board_id:str,body:ComponentsUpdate):
    board = store.board(board_id)
    board['components'] = confirm_components(board,[c.model_dump() for c in body.components])
    board['confirmed'] = True
    store.save_board(board)
    (store.DATA/'boards'/board['id']/'profile.json').write_text(json.dumps(board,indent=2),encoding='utf-8')
    return board

@app.post('/api/boards/{board_id}/analyze',status_code=202)
async def new_analysis(board_id:str,request:Request,test:Annotated[UploadFile,File()],debug:bool=False):
    board = store.board(board_id)
    if not board['confirmed']: raise AppError('profile_incomplete','Confirm and save component boxes before analyzing a test photo.')
    raw = await read_upload(test,IMAGE_EXTENSIONS)
    key = hashlib.sha256(raw+json.dumps(board,sort_keys=True).encode()+json.dumps(CONFIG,sort_keys=True).encode()+str(debug).encode()).hexdigest()
    analysis_id = uuid.uuid4().hex
    with store.connection() as db:
        # A transaction serializes cache lookup and queue admission.
        db.execute('BEGIN IMMEDIATE')
        cached = db.execute('SELECT id,status FROM analyses WHERE cache_key=?',(key,)).fetchone()
        if cached and cached['status']!='failed': return {'analysis_id':cached['id'],'cached':True}
        pending = db.execute("SELECT count(*) FROM analyses WHERE status IN ('queued','running')").fetchone()[0]
        if pending>=CONFIG['runtime']['max_pending']: raise AppError('busy','The analysis queue is full. Try again shortly.',429)
        if cached: db.execute('DELETE FROM analyses WHERE id=?',(cached['id'],))
        db.execute('INSERT INTO analyses (id,board_id,cache_key,status) VALUES (?,?,?,?)',(analysis_id,board_id,key,'queued'))
    request.app.state.executor.submit(analyze,analysis_id,board,raw,debug)
    return {'analysis_id':analysis_id,'cached':False}

@app.get('/api/analyses/{analysis_id}')
def get_analysis(analysis_id:str):
    entry = store.analysis(analysis_id)
    return {'analysis_id':analysis_id,'status':entry['status'],'error':entry['error'],**(entry['result'] or {})}

def completed(analysis_id):
    entry = store.analysis(analysis_id)
    if entry['status']!='completed': raise AppError('analysis_not_ready','Analysis has not completed successfully.',409)
    return entry['result'],store.DATA/'analyses'/entry['id']

@app.get('/api/analyses/{analysis_id}/source')
def analysis_source(analysis_id:str,rectified:bool=False):
    report,directory = completed(analysis_id)
    return FileResponse(directory/('rectified.png' if rectified else 'original.png'),media_type='image/png')

@app.get('/api/analyses/{analysis_id}/image')
def analysis_image(analysis_id:str,rectified:bool=False,show_present:bool=False):
    report,directory = completed(analysis_id)
    image = cv2.imread(str(directory/('rectified.png' if rectified else 'original.png')))
    encoded = cv2.imencode('.png',annotate(image,report['components'],rectified,show_present))[1].tobytes()
    return Response(encoded,media_type='image/png',headers={'Content-Disposition':'attachment; filename="pcb-annotated.png"'})

@app.get('/api/analyses/{analysis_id}/report')
def analysis_report(analysis_id:str,format:Literal['json','pdf']='json'):
    report,directory = completed(analysis_id)
    if format=='json': return FileResponse(directory/'report.json',media_type='application/json',filename='pcb-report.json')
    return Response(pdf_report(report,directory/'annotated.png'),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename="pcb-report.pdf"'})

@app.post('/api/demo',status_code=201)
def demo_board():
    from backend.vision.synthetic import fixture, PARTS
    raw = cv2.imencode('.png',fixture())[1].tobytes()
    parts = parse_parts('schematic.net',(ROOT/'fixtures/ne555/schematic.net').read_bytes())
    rois = {p['designator']:p['roi'] for p in PARTS}
    for part in parts: part['roi'] = rois[part['designator']]
    board = create_profile('NE555 timer · demo',raw,parts,86.25,61.25)
    # The known synthetic fixture supplies exact annotations.
    board['components'] = confirm_components(board,parts)
    board['confirmed'] = True
    store.save_board(board)
    return board

@app.get('/api/demo/test')
def demo_test():
    return FileResponse(ROOT/'fixtures/ne555/test_missing.png',media_type='image/png',filename='test_missing.png')

app.mount('/',StaticFiles(directory=ROOT/'frontend',html=True,check_dir=False),name='frontend')
