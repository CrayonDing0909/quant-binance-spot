"""
post_dca_social 的最小測試。

刻意設計成「不碰網路、不需要 qtrade/vectorbt」：
  - 報告來源 (get_report_text) 與 HTTP (requests) 都被 mock 掉。
  - 只驗證憑證讀取、Threads 字數處理、圖片渲染、dry-run 預覽、以及兩步發文流程。

執行：PYTHONPATH 由 `python -m pytest` 自動補上 repo root。
"""
from __future__ import annotations

import sys
import types

import pytest

from scripts.post_dca_social import (
    SocialCredentials,
    condense_report_for_threads,
    find_cjk_font,
    get_report_text,
    load_credentials,
    main,
    post_instagram,
    post_threads,
    render_report_image,
    send_telegram,
    truncate_caption,
    THREADS_TEXT_LIMIT,
)

SAMPLE_REPORT = """因為判斷為相對底部了, 所以持續購入BTC 與ETH Day5
-
定投報告
累積投入: 500.00U
目前市值: 540.00U
未實現損益: +40.00U (+8.00%)

持倉明細
BTCUSDT:
  投入金額: 250.00U
  未實現損益: +20.00U (+8.00%)
ETHUSDT:
  投入金額: 250.00U
  未實現損益: +20.00U (+8.00%)"""


# ── 假的 requests，攔截 HTTP 呼叫 ──────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, post_responses=None, get_responses=None):
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    def post(self, url, data=None, files=None, params=None, timeout=None):
        self.posts.append({"url": url, "data": data or {}, "files": files, "params": params})
        return self.post_responses.pop(0)

    def get(self, url, params=None, timeout=None):
        self.gets.append({"url": url, "params": params or {}})
        return self.get_responses.pop(0)


@pytest.fixture
def fake_requests(monkeypatch):
    def _install(post_responses=None, get_responses=None):
        fake = _FakeRequests(post_responses, get_responses)
        module = types.ModuleType("requests")
        module.post = fake.post
        module.get = fake.get
        monkeypatch.setitem(sys.modules, "requests", module)
        return fake

    return _install


# ── 憑證 ──────────────────────────────────────────────────────────────────

def test_load_credentials_threads_token_falls_back_to_meta(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "meta-tok")
    monkeypatch.setenv("IG_USER_ID", "ig-123")
    monkeypatch.setenv("THREADS_USER_ID", "th-456")
    monkeypatch.delenv("THREADS_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-bot")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "tg-chat")

    creds = load_credentials()
    assert creds.meta_access_token == "meta-tok"
    assert creds.threads_access_token == "meta-tok"  # fallback
    assert creds.instagram_ready is True
    assert creds.threads_ready is True
    assert creds.telegram_ready is True


def test_load_credentials_separate_threads_token(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "meta-tok")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "threads-tok")
    monkeypatch.delenv("IG_USER_ID", raising=False)
    monkeypatch.delenv("THREADS_USER_ID", raising=False)

    creds = load_credentials()
    assert creds.threads_access_token == "threads-tok"
    assert creds.instagram_ready is False  # 缺 IG_USER_ID
    assert creds.threads_ready is False     # 缺 THREADS_USER_ID


# ── Threads 字數處理 ──────────────────────────────────────────────────────

def test_condense_under_limit_returns_verbatim():
    assert condense_report_for_threads(SAMPLE_REPORT) == SAMPLE_REPORT


def test_condense_over_limit_fits_and_keeps_key_lines():
    long_report = SAMPLE_REPORT + "\n" + ("補充說明 " * 200)
    out = condense_report_for_threads(long_report, limit=THREADS_TEXT_LIMIT)
    assert len(out) <= THREADS_TEXT_LIMIT
    assert "持續購入" in out
    assert "累積投入: 500.00U" in out


def test_truncate_caption():
    assert truncate_caption("x" * 10, limit=5) == "xxxx…"
    assert truncate_caption("hello", limit=5) == "hello"


# ── 字型探測 ──────────────────────────────────────────────────────────────

def test_find_cjk_font_respects_override(monkeypatch, tmp_path):
    fake_font = tmp_path / "myfont.ttf"
    fake_font.write_bytes(b"not-a-real-font")
    monkeypatch.setenv("DCA_REPORT_FONT", str(fake_font))
    assert find_cjk_font() == str(fake_font)


# ── 圖片渲染 ──────────────────────────────────────────────────────────────

def test_render_report_image_writes_valid_jpeg(tmp_path):
    out = tmp_path / "report.jpg"
    path = render_report_image(SAMPLE_REPORT, out, image_format="jpg")
    assert path.exists() and path.stat().st_size > 0

    from PIL import Image
    img = Image.open(path)
    assert img.size == (1080, 1350)
    assert img.format == "JPEG"


# ── dry-run / 降級行為（不需網路）────────────────────────────────────────

def test_post_threads_dry_run_no_network(capsys):
    creds = SocialCredentials(None, None, None, None)
    res = post_threads(creds, "嗨", dry_run=True)
    assert res["dry_run"] is True and res["channel"] == "threads"
    assert "嗨" in capsys.readouterr().out


def test_post_threads_missing_creds_degrades_to_dry_run():
    creds = SocialCredentials(None, None, None, None)  # threads_ready False
    res = post_threads(creds, "嗨", dry_run=False)
    assert res["dry_run"] is True  # 降級，不會炸


def test_post_instagram_dry_run_no_network(capsys):
    creds = SocialCredentials(None, None, None, None)
    res = post_instagram(creds, "https://x/y.jpg", "caption", dry_run=True)
    assert res["dry_run"] is True and res["channel"] == "instagram"
    assert "https://x/y.jpg" in capsys.readouterr().out


def test_send_telegram_dry_run_no_network(capsys):
    creds = SocialCredentials(None, None, None, None)
    res = send_telegram(creds, "今日文案", dry_run=True)
    assert res["dry_run"] is True and res["channel"] == "telegram"
    assert "今日文案" in capsys.readouterr().out


def test_send_telegram_missing_creds_degrades():
    creds = SocialCredentials(None, None, None, None)  # telegram_ready False
    res = send_telegram(creds, "x", dry_run=False)
    assert res["dry_run"] is True


def test_send_telegram_live_calls_api(fake_requests):
    fake = fake_requests(post_responses=[_FakeResp({"result": {"message_id": 999}})])
    creds = SocialCredentials(None, None, None, None,
                              telegram_bot_token="botT", telegram_chat_id="42")
    res = send_telegram(creds, "貼文文案", dry_run=False)
    assert res["id"] == 999 and res["dry_run"] is False
    assert "/botbotT/sendMessage" in fake.posts[0]["url"]
    assert fake.posts[0]["data"]["chat_id"] == "42"
    assert fake.posts[0]["data"]["text"] == "貼文文案"


# ── 兩步發文流程（mock requests）──────────────────────────────────────────

def test_post_threads_live_two_step_flow(fake_requests):
    fake = fake_requests(post_responses=[_FakeResp({"id": "CREATE1"}), _FakeResp({"id": "MEDIA1"})])
    creds = SocialCredentials("meta", "ig", "th-user", "th-tok")

    res = post_threads(creds, "貼文內容", dry_run=False)

    assert res["id"] == "MEDIA1" and res["dry_run"] is False
    assert fake.posts[0]["url"].endswith("/th-user/threads")
    assert fake.posts[0]["data"]["media_type"] == "TEXT"
    assert fake.posts[0]["data"]["text"] == "貼文內容"
    assert fake.posts[1]["url"].endswith("/th-user/threads_publish")
    assert fake.posts[1]["data"]["creation_id"] == "CREATE1"


def test_post_instagram_live_polls_then_publishes(fake_requests):
    fake = fake_requests(
        post_responses=[_FakeResp({"id": "IGC1"}), _FakeResp({"id": "IGM1"})],
        get_responses=[_FakeResp({"status_code": "FINISHED"})],
    )
    creds = SocialCredentials("meta-tok", "ig-user", "th", "th")

    res = post_instagram(creds, "https://pub/img.jpg", "說明", dry_run=False, wait_seconds=0)

    assert res["id"] == "IGM1" and res["dry_run"] is False
    assert fake.posts[0]["url"].endswith("/ig-user/media")
    assert fake.posts[0]["data"]["image_url"] == "https://pub/img.jpg"
    assert fake.gets[0]["params"]["fields"].startswith("status_code")
    assert fake.posts[1]["url"].endswith("/ig-user/media_publish")
    assert fake.posts[1]["data"]["creation_id"] == "IGC1"


def test_post_instagram_live_raises_on_container_error(fake_requests):
    fake_requests(
        post_responses=[_FakeResp({"id": "IGC1"})],
        get_responses=[_FakeResp({"status_code": "ERROR"})],
    )
    creds = SocialCredentials("meta-tok", "ig-user", "th", "th")
    with pytest.raises(RuntimeError):
        post_instagram(creds, "https://pub/img.jpg", "說明", dry_run=False, wait_seconds=0)


# ── 報告擷取（mock subprocess）────────────────────────────────────────────

def test_get_report_text_returns_stdout(monkeypatch):
    import subprocess

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        return types.SimpleNamespace(returncode=0, stdout=SAMPLE_REPORT + "\n", stderr="log noise")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert get_report_text("config/dca.yaml") == SAMPLE_REPORT


def test_get_report_text_raises_on_failure(monkeypatch):
    import subprocess

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        get_report_text("config/dca.yaml")


# ── main dry-run 整合（無網路、無 token）──────────────────────────────────

def test_main_dry_run_previews_all_channels(monkeypatch, tmp_path, capsys):
    import scripts.post_dca_social as mod

    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(mod, "get_report_text", lambda *a, **k: SAMPLE_REPORT)
    for var in ("META_ACCESS_TOKEN", "IG_USER_ID", "THREADS_USER_ID", "THREADS_ACCESS_TOKEN",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)

    rc = main(["--dry-run", "--image-out", str(tmp_path / "r.jpg")])
    out = capsys.readouterr().out

    assert rc == 0
    assert "定投報告" in out
    assert "[Telegram]" in out
    assert "[Threads]" in out
    assert "[Instagram]" in out
    assert (tmp_path / "r.jpg").exists()  # 圖片仍會被產生


def test_main_send_telegram_only_skips_other_channels(monkeypatch, tmp_path, capsys):
    import scripts.post_dca_social as mod

    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(mod, "get_report_text", lambda *a, **k: SAMPLE_REPORT)
    for var in ("META_ACCESS_TOKEN", "IG_USER_ID", "THREADS_USER_ID", "THREADS_ACCESS_TOKEN",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)

    img = tmp_path / "nope.jpg"
    rc = main(["--send-telegram", "--dry-run", "--image-out", str(img)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "[Telegram]" in out
    assert "[Threads]" not in out
    assert "[Instagram]" not in out
    assert not img.exists()  # 沒選 IG → 不渲染圖片
