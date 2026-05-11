from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

DB_PATH = Path("data/voc.db")


@dataclass
class Post:
    post_id: str
    url: str
    title: str = ""
    author: str = ""
    created_at: str = ""
    board: str = ""
    content: str = ""
    keyword: str = ""
    fetched_at: str = ""


@dataclass
class Reply:
    reply_id: str
    post_id: str
    url: str = ""
    author: str = ""
    role: str = ""
    created_at: str = ""
    content: str = ""
    is_moderator: int = 0


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                author TEXT,
                created_at TEXT,
                board TEXT,
                content TEXT,
                keyword TEXT,
                fetched_at TEXT
            );

            CREATE TABLE IF NOT EXISTS replies (
                reply_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                url TEXT,
                author TEXT,
                role TEXT,
                created_at TEXT,
                content TEXT,
                is_moderator INTEGER DEFAULT 0,
                FOREIGN KEY(post_id) REFERENCES posts(post_id)
            );

            CREATE INDEX IF NOT EXISTS idx_posts_keyword ON posts(keyword);
            CREATE INDEX IF NOT EXISTS idx_replies_post_id ON replies(post_id);
            CREATE INDEX IF NOT EXISTS idx_replies_moderator ON replies(is_moderator);
            """
        )


def upsert_post(post: Post, db_path: Path | str = DB_PATH) -> None:
    init_db(db_path)
    data = asdict(post)
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO posts(post_id, url, title, author, created_at, board, content, keyword, fetched_at)
            VALUES(:post_id, :url, :title, :author, :created_at, :board, :content, :keyword, :fetched_at)
            ON CONFLICT(post_id) DO UPDATE SET
                url=excluded.url,
                title=excluded.title,
                author=excluded.author,
                created_at=excluded.created_at,
                board=excluded.board,
                content=excluded.content,
                keyword=excluded.keyword,
                fetched_at=excluded.fetched_at
            """,
            data,
        )


def upsert_replies(replies: Iterable[Reply], db_path: Path | str = DB_PATH) -> None:
    init_db(db_path)
    rows = [asdict(r) for r in replies]
    if not rows:
        return
    with connect(db_path) as con:
        con.executemany(
            """
            INSERT INTO replies(reply_id, post_id, url, author, role, created_at, content, is_moderator)
            VALUES(:reply_id, :post_id, :url, :author, :role, :created_at, :content, :is_moderator)
            ON CONFLICT(reply_id) DO UPDATE SET
                post_id=excluded.post_id,
                url=excluded.url,
                author=excluded.author,
                role=excluded.role,
                created_at=excluded.created_at,
                content=excluded.content,
                is_moderator=excluded.is_moderator
            """,
            rows,
        )


def fetch_posts(keyword: Optional[str] = None, db_path: Path | str = DB_PATH):
    init_db(db_path)
    with connect(db_path) as con:
        if keyword:
            return con.execute(
                "SELECT * FROM posts WHERE keyword LIKE ? OR title LIKE ? OR content LIKE ? ORDER BY fetched_at DESC",
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
            ).fetchall()
        return con.execute("SELECT * FROM posts ORDER BY fetched_at DESC").fetchall()


def fetch_replies(post_id: Optional[str] = None, only_moderator: bool = False, db_path: Path | str = DB_PATH):
    init_db(db_path)
    clauses = []
    params = []
    if post_id:
        clauses.append("post_id = ?")
        params.append(post_id)
    if only_moderator:
        clauses.append("is_moderator = 1")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with connect(db_path) as con:
        return con.execute(f"SELECT * FROM replies{where} ORDER BY created_at DESC", params).fetchall()


def export_joined(db_path: Path | str = DB_PATH):
    init_db(db_path)
    with connect(db_path) as con:
        return con.execute(
            """
            SELECT
                p.post_id, p.keyword, p.board, p.title, p.author AS post_author,
                p.created_at AS post_created_at, p.url AS post_url, p.content AS post_content,
                r.reply_id, r.author AS reply_author, r.role AS reply_role,
                r.created_at AS reply_created_at, r.content AS moderator_reply
            FROM posts p
            LEFT JOIN replies r ON p.post_id = r.post_id AND r.is_moderator = 1
            ORDER BY p.fetched_at DESC
            """
        ).fetchall()


def clear_db(db_path: Path | str = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as con:
        con.execute("DELETE FROM replies")
        con.execute("DELETE FROM posts")
