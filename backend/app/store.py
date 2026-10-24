import json
import sqlite3
from contextlib import contextmanager
from backend.app.config import DATA
from backend.app.errors import AppError

@contextmanager
def connection():
    DATA.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATA/'pcb.sqlite3', timeout=30) as db:
        db.row_factory = sqlite3.Row
        yield db

def init():
    with connection() as db:
        db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS boards (id TEXT PRIMARY KEY, profile TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, board_id TEXT NOT NULL,
                cache_key TEXT UNIQUE NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        ''')
        db.execute("UPDATE analyses SET status='failed', error=? WHERE status IN ('queued','running')",
                   (json.dumps(dict(code='interrupted',message='Analysis was interrupted by a server restart. Upload the photo again.')),))

def save_board(profile):
    with connection() as db:
        db.execute('INSERT INTO boards VALUES (?,?) ON CONFLICT(id) DO UPDATE SET profile=excluded.profile',
                   (profile['id'],json.dumps(profile)))

def board(board_id):
    with connection() as db:
        row = db.execute('SELECT profile FROM boards WHERE id=?',(board_id,)).fetchone()
    if not row: raise AppError('board_not_found','Board profile not found.',404)
    return json.loads(row['profile'])

def boards():
    with connection() as db:
        return [json.loads(row['profile']) for row in db.execute('SELECT profile FROM boards ORDER BY rowid DESC')]

def analysis(analysis_id):
    with connection() as db:
        row = db.execute('SELECT * FROM analyses WHERE id=?',(analysis_id,)).fetchone()
    if not row: raise AppError('analysis_not_found','Analysis not found.',404)
    result = dict(row)
    result['result'] = json.loads(result['result']) if result['result'] else None
    result['error'] = json.loads(result['error']) if result['error'] else None
    return result

def update(analysis_id, status, result=None, error=None):
    with connection() as db:
        db.execute('UPDATE analyses SET status=?,result=?,error=? WHERE id=?',
                   (status,json.dumps(result) if result else None,json.dumps(error) if error else None,analysis_id))
