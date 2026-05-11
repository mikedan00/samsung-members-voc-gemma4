from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from .db import Post, Reply, upsert_post, upsert_replies

BASE = "https://r1.community.samsung.com"
SEARCH_URL = f"{BASE}/t5/forums/searchpage/tab/message"
DEFAULT_UA = os.getenv(
    "SAMSUNG_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/147.0 Safari/537.36",
)

POST_ID_RE = re.compile(r"/(?:m-p|td-p)/(\d+)")
HTML_SPACE_RE = re.compile(r"\s+")


@dataclass
class CrawlResult:
    keyword: str
    discovered_urls: int
    saved_posts: int
    saved_replies: int
    errors: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\u200e", " ").replace("\xa0", " ")
    return HTML_SPACE_RE.sub(" ", text).strip()


def normalize_url(href: str) -> str:
    url = urljoin(BASE, href)
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def post_id_from_url(url: str) -> str:
    m = POST_ID_RE.search(url)
    if m:
        return m.group(1)
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def make_search_url(keyword: str, page: int = 1) -> str:
    # 삼성멤버스 검색 URL 구조를 기준으로 생성합니다.
    q = quote_plus(keyword)
    return (
        f"{SEARCH_URL}?advanced=false&allow_punctuation=false&filter=location"
        f"&location=category:kr-community&q={q}&page={page}"
    )


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": DEFAULT_UA,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )
    return s


def fetch_html(url: str, session: Optional[requests.Session] = None, timeout: int = 20) -> str:
    s = session or get_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_post_links(search_html: str, max_links: int = 30) -> list[str]:
    soup = BeautifulSoup(search_html, "lxml")
    links: list[str] = []
    seen: set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        if "/m-p/" not in href and "/td-p/" not in href:
            continue
        url = normalize_url(href)
        if url in seen:
            continue
        seen.add(url)
        links.append(url)
        if len(links) >= max_links:
            break
    return links


def first_text(soup_or_node, selectors: Iterable[str]) -> str:
    for selector in selectors:
        node = soup_or_node.select_one(selector)
        if node:
            txt = clean_text(node.get_text(" ", strip=True))
            if txt:
                return txt
    return ""


def first_attr(soup: BeautifulSoup, selectors: Iterable[tuple[str, str]]) -> str:
    for selector, attr in selectors:
        node = soup.select_one(selector)
        if node and node.get(attr):
            return clean_text(node.get(attr))
    return ""


def extract_jsonld_article(soup: BeautifulSoup) -> dict:
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            t = item.get("@type")
            if isinstance(t, list):
                is_article = any(str(x).lower() in {"article", "discussionforumpost"} for x in t)
            else:
                is_article = str(t).lower() in {"article", "discussionforumpost", "creativework"}
            if is_article or item.get("headline") or item.get("articleBody"):
                return item
    return {}


def extract_title(soup: BeautifulSoup, article: dict) -> str:
    if article.get("headline"):
        return clean_text(article.get("headline"))
    title = first_attr(
        soup,
        [
            ('meta[property="og:title"]', "content"),
            ('meta[name="twitter:title"]', "content"),
        ],
    )
    if title:
        return title
    title = first_text(
        soup,
        [
            "h1",
            ".lia-message-subject",
            ".message-subject",
            ".lia-thread-topic-title",
            "title",
        ],
    )
    return title.replace(" - Samsung Members", "").strip()


def extract_author(soup: BeautifulSoup, article: dict) -> str:
    author = article.get("author")
    if isinstance(author, dict):
        return clean_text(author.get("name", ""))
    if isinstance(author, str):
        return clean_text(author)
    return first_text(
        soup,
        [
            ".lia-user-name",
            ".lia-message-author-with-avatar .lia-user-name",
            ".UserName",
            '[itemprop="author"]',
        ],
    )


def extract_created_at(soup: BeautifulSoup, article: dict) -> str:
    for key in ("datePublished", "dateCreated", "dateModified"):
        if article.get(key):
            return clean_text(article[key])
    return first_attr(
        soup,
        [
            ("time", "datetime"),
            ('meta[property="article:published_time"]', "content"),
            ('meta[itemprop="datePublished"]', "content"),
        ],
    ) or first_text(soup, [".local-date", ".DateTime", ".lia-message-post-date"])


def extract_board(soup: BeautifulSoup) -> str:
    return first_text(
        soup,
        [
            ".lia-breadcrumb .lia-list-standard-inline li:last-child",
            ".lia-link-navigation.breadcrumb-link:last-child",
            ".lia-component-common-widget-breadcrumb",
        ],
    )


def extract_content(soup: BeautifulSoup, article: dict) -> str:
    if article.get("articleBody"):
        return clean_text(article.get("articleBody"))
    content = first_text(
        soup,
        [
            ".lia-message-body-content",
            ".lia-message-body",
            '[itemprop="text"]',
            ".message-body",
            "article",
        ],
    )
    if content:
        return content
    # 마지막 fallback: nav/footer를 제외한 body 텍스트 일부
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return clean_text(soup.get_text(" ", strip=True))[:8000]


def is_moderator(author: str, role: str, content: str, block_text: str = "") -> bool:
    hay = f"{author} {role} {content[:300]} {block_text[:500]}".lower()
    korean_hay = f"{author} {role} {content[:300]} {block_text[:500]}"
    if "moderator" in hay or "admin" in hay:
        return True
    if "담당" in korean_hay or "운영자" in korean_hay or "관리자" in korean_hay:
        return True
    return False


def extract_message_blocks(soup: BeautifulSoup):
    selectors = [
        '[id^="message-uid"]',
        ".lia-message-view",
        ".lia-quilt-forum-message",
        ".lia-linear-display-message-view",
        "article",
    ]
    blocks = []
    seen = set()
    for selector in selectors:
        for node in soup.select(selector):
            txt = clean_text(node.get_text(" ", strip=True))
            if len(txt) < 30:
                continue
            key = hashlib.sha1(txt[:500].encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            blocks.append(node)
        if len(blocks) >= 2:
            break
    return blocks


def extract_replies(soup: BeautifulSoup, post_id: str, post_url: str, main_content: str) -> list[Reply]:
    replies: list[Reply] = []
    blocks = extract_message_blocks(soup)
    for idx, block in enumerate(blocks):
        content = first_text(
            block,
            [
                ".lia-message-body-content",
                ".lia-message-body",
                '[itemprop="text"]',
                ".message-body",
            ],
        ) or clean_text(block.get_text(" ", strip=True))

        # 첫 본문과 거의 같은 블록은 원글로 보고 제외합니다.
        if idx == 0 and main_content and content[:300] == main_content[:300]:
            continue
        if len(content) < 20:
            continue

        author = first_text(
            block,
            [
                ".lia-user-name",
                ".lia-message-author-with-avatar .lia-user-name",
                ".UserName",
                '[itemprop="author"]',
            ],
        )
        role = first_text(block, [".lia-user-rank", ".lia-user-role", ".lia-message-author-rank", ".rank-name"])
        created_at = first_attr(block, [("time", "datetime")]) or first_text(
            block, [".local-date", ".DateTime", ".lia-message-post-date"]
        )
        block_text = clean_text(block.get_text(" ", strip=True))
        mod = is_moderator(author, role, content, block_text)

        # 댓글 URL은 상세 anchor가 있으면 사용합니다.
        reply_url = post_url
        a = block.select_one('a[href*="/m-p/"], a[href*="/td-p/"]')
        if a and a.get("href"):
            reply_url = normalize_url(a["href"])

        rid_src = f"{post_id}|{idx}|{author}|{created_at}|{content[:120]}"
        reply_id = hashlib.sha1(rid_src.encode("utf-8")).hexdigest()[:20]
        replies.append(
            Reply(
                reply_id=reply_id,
                post_id=post_id,
                url=reply_url,
                author=author,
                role=role,
                created_at=created_at,
                content=content,
                is_moderator=1 if mod else 0,
            )
        )
    return replies


def parse_post(url: str, html: str, keyword: str = "") -> tuple[Post, list[Reply]]:
    soup = BeautifulSoup(html, "lxml")
    article = extract_jsonld_article(soup)
    post_id = post_id_from_url(url)
    content = extract_content(soup, article)
    post = Post(
        post_id=post_id,
        url=normalize_url(url),
        title=extract_title(soup, article),
        author=extract_author(soup, article),
        created_at=extract_created_at(soup, article),
        board=extract_board(soup),
        content=content,
        keyword=keyword,
        fetched_at=now_iso(),
    )
    replies = extract_replies(soup, post_id, post.url, content)
    return post, replies


def crawl_keyword(
    keyword: str,
    pages: int = 1,
    max_posts: int = 20,
    delay_sec: float = 1.0,
    db_path: str = "data/voc.db",
) -> CrawlResult:
    session = get_session()
    urls: list[str] = []
    errors: list[str] = []

    for page in range(1, pages + 1):
        search_url = make_search_url(keyword, page)
        try:
            html = fetch_html(search_url, session=session)
            urls.extend(extract_post_links(html, max_links=max_posts))
        except Exception as e:
            errors.append(f"검색 페이지 수집 실패 page={page}: {e}")
        time.sleep(max(delay_sec, 0.2))

    # 중복 제거 및 개수 제한
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    unique_urls = unique_urls[:max_posts]

    saved_posts = 0
    saved_replies = 0
    for url in unique_urls:
        try:
            html = fetch_html(url, session=session)
            post, replies = parse_post(url, html, keyword=keyword)
            upsert_post(post, db_path=db_path)
            upsert_replies(replies, db_path=db_path)
            saved_posts += 1
            saved_replies += len(replies)
        except Exception as e:
            errors.append(f"게시글 수집 실패 url={url}: {e}")
        time.sleep(max(delay_sec, 0.2))

    return CrawlResult(
        keyword=keyword,
        discovered_urls=len(unique_urls),
        saved_posts=saved_posts,
        saved_replies=saved_replies,
        errors=errors,
    )


def import_url_list(urls: Iterable[str], keyword: str = "manual", delay_sec: float = 1.0, db_path: str = "data/voc.db") -> CrawlResult:
    session = get_session()
    saved_posts = 0
    saved_replies = 0
    errors: list[str] = []
    clean_urls = [normalize_url(u.strip()) for u in urls if u and u.strip()]
    for url in clean_urls:
        try:
            html = fetch_html(url, session=session)
            post, replies = parse_post(url, html, keyword=keyword)
            upsert_post(post, db_path=db_path)
            upsert_replies(replies, db_path=db_path)
            saved_posts += 1
            saved_replies += len(replies)
        except Exception as e:
            errors.append(f"URL 수집 실패 url={url}: {e}")
        time.sleep(max(delay_sec, 0.2))
    return CrawlResult(keyword=keyword, discovered_urls=len(clean_urls), saved_posts=saved_posts, saved_replies=saved_replies, errors=errors)
