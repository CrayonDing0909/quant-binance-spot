#!/usr/bin/env python3
"""
DCA 社群自動發文 — Instagram + Threads。

把 `scripts/run_dca.py --report` 產生的中文定投報告，自動發到：
  - Threads：純文字貼文（graph.threads.net）
  - Instagram：因為 IG 不能只發純文字，先把報告渲染成一張中文圖片，
    上傳到「可公開讀取」的位置，再用 Instagram Content Publishing API 發圖文。

安全設計：
  - 所有 token / user id 都從 .env（或環境變數）讀取，絕不寫進 repo。
  - 預設 dry-run：不帶 --post-threads / --post-instagram 時，只印出「將會發什麼」。
  - 缺少憑證時不會炸掉，而是該頻道自動降級成 dry-run 並印出警告，
    這樣排程現在就能安裝，等之後補上 token 自然就會真的發文。

範例：
    # 只預覽（不發文、不需要任何 token）
    python scripts/post_dca_social.py -c config/dca.yaml --dry-run

    # 真的發文（需要 .env 內的 token）
    python scripts/post_dca_social.py -c config/dca.yaml --post-threads --post-instagram

憑證（.env，見 .env.example）：
    META_ACCESS_TOKEN      Instagram Graph API 用的長效 token
    IG_USER_ID             Instagram 商業帳號 user id
    THREADS_USER_ID        Threads 帳號 user id
    THREADS_ACCESS_TOKEN   Threads API token（Threads 與 IG 通常是「不同」的 token；
                           若未設定會 fallback 用 META_ACCESS_TOKEN）
    IMGBB_API_KEY 或 PUBLIC_IMAGE_BASE_URL / PUBLIC_IMAGE_DIR  IG 圖片公開託管
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DCA = REPO_ROOT / "scripts" / "run_dca.py"

# Threads 單篇純文字上限是 500 字；超過就改發精簡版（並保底截斷）。
THREADS_TEXT_LIMIT = 500
# Instagram caption 上限 2200 字。
IG_CAPTION_LIMIT = 2200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s post_dca_social - %(message)s",
    stream=sys.stderr,  # log 走 stderr，stdout 保留給報告預覽
)
logger = logging.getLogger("post_dca_social")


# ── 憑證 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SocialCredentials:
    meta_access_token: str | None
    ig_user_id: str | None
    threads_user_id: str | None
    threads_access_token: str | None

    @property
    def threads_ready(self) -> bool:
        return bool(self.threads_user_id and self.threads_access_token)

    @property
    def instagram_ready(self) -> bool:
        return bool(self.meta_access_token and self.ig_user_id)


def load_credentials() -> SocialCredentials:
    """從環境變數讀取憑證。Threads token 若未提供則 fallback 用 META_ACCESS_TOKEN。"""
    meta = os.getenv("META_ACCESS_TOKEN") or None
    # Threads 與 Instagram 通常是兩個不同的 token / 不同的 OAuth。
    # 但若使用者只給一個，就讓 Threads 退而求其次用 META_ACCESS_TOKEN，避免卡住。
    threads_token = os.getenv("THREADS_ACCESS_TOKEN") or meta
    return SocialCredentials(
        meta_access_token=meta,
        ig_user_id=os.getenv("IG_USER_ID") or None,
        threads_user_id=os.getenv("THREADS_USER_ID") or None,
        threads_access_token=threads_token,
    )


# ── 取得報告文字 ──────────────────────────────────────────────────────────

def get_report_text(config_path: str, python_exe: str | None = None) -> str:
    """
    呼叫 `run_dca.py --report` 取得中文報告。

    用 subprocess 是刻意的：run_dca 會 import 整個 qtrade（vectorbt 等）重依賴，
    這裡用子行程隔離；報告走 stdout、log 走 stderr，所以 stdout 乾淨。
    """
    python_exe = python_exe or sys.executable
    cmd = [python_exe, str(RUN_DCA), "-c", config_path, "--report"]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_dca.py --report 失敗 (code={proc.returncode}):\n{proc.stderr.strip()}"
        )
    report = proc.stdout.strip()
    if not report:
        raise RuntimeError(f"run_dca.py --report 沒有輸出任何內容\nstderr:\n{proc.stderr.strip()}")
    return report


# ── 文字處理 ──────────────────────────────────────────────────────────────

def condense_report_for_threads(report: str, limit: int = THREADS_TEXT_LIMIT) -> str:
    """
    Threads 純文字上限 500 字。報告若沒超過就原樣回傳；
    超過就只保留「標題 + 總結（投入/市值/損益）+ 各標的損益%」這幾行，仍超過才硬截斷。
    """
    if len(report) <= limit:
        return report

    lines = report.splitlines()
    keep: list[str] = []
    for line in lines:
        stripped = line.strip()
        # 保留：開頭敘述、總結三行、以及每個標的的損益百分比行。
        if (
            "持續購入" in stripped
            or stripped.startswith(("累積投入", "目前市值", "未實現損益"))
            or stripped.endswith(("USDT:", "USD:"))
        ):
            keep.append(stripped)
    condensed = "\n".join(keep).strip() or report
    if len(condensed) > limit:
        condensed = condensed[: limit - 1].rstrip() + "…"
    return condensed


def truncate_caption(text: str, limit: int = IG_CAPTION_LIMIT) -> str:
    """IG caption 上限保護。"""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ── 報告圖片渲染（給 Instagram）──────────────────────────────────────────

# CJK 字型候選：macOS（本機）+ Linux（Oracle VM）。可用 DCA_REPORT_FONT 覆寫。
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux（Ubuntu / Oracle Cloud）常見 CJK 字型
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def find_cjk_font() -> str | None:
    """回傳第一個存在的 CJK 字型路徑；找不到回傳 None（呼叫端需處理 fallback）。"""
    override = os.getenv("DCA_REPORT_FONT")
    candidates = ([override] if override else []) + _FONT_CANDIDATES
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def render_report_image(
    report: str,
    out_path: str | Path,
    *,
    image_format: str = "jpg",
    width: int = 1080,
    height: int = 1350,
    font_path: str | None = None,
) -> Path:
    """
    把報告文字渲染成一張深色報表卡片。

    預設 1080x1350（IG 允許的 4:5 直式上限），輸出 JPEG（IG 對 JPEG 支援最穩）。
    PIL 為延遲匯入，模組本身不硬性依賴 Pillow。
    """
    from PIL import Image, ImageDraw, ImageFont  # 延遲匯入

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg = (13, 17, 23)        # GitHub 深色底
    fg = (230, 237, 243)
    accent = (88, 166, 255)
    muted = (139, 148, 158)

    font_path = font_path or find_cjk_font()
    lines = report.splitlines()

    def load_font(size: int):
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception as exc:  # 字型損毀等
                logger.warning("載入字型 %s 失敗：%s", font_path, exc)
        return ImageFont.load_default()

    if not font_path:
        logger.warning(
            "找不到 CJK 字型，中文可能變成方框。請安裝 Noto Sans CJK 或設定 DCA_REPORT_FONT。"
        )

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    margin = 72
    title_size = 52
    footer_size = 24
    footer_reserve = footer_size + 36  # 預留頁尾空間，避免內文壓到頁尾
    line_gap = 14
    usable_h = height - margin * 2 - (title_size + 40) - footer_reserve
    usable_w = width - margin * 2
    longest = max(lines, key=len) if lines else ""

    # 依「行數(高)」與「最長行(寬)」自動縮放 body 字級，確保不溢出也不壓到頁尾。
    body_size = 34
    while body_size > 16:
        line_h = body_size + line_gap
        fits_h = len(lines) * line_h <= usable_h
        fits_w = draw.textlength(longest, font=load_font(body_size)) <= usable_w
        if fits_h and fits_w:
            break
        body_size -= 2
    line_h = body_size + line_gap

    title_font = load_font(title_size)
    body_font = load_font(body_size)

    y = margin
    draw.text((margin, y), "BTC / ETH 定投報告", font=title_font, fill=accent)
    y += title_size + 40

    for line in lines:
        # 第一行（敘述）與分隔線用 muted 色，其餘用前景色。
        color = muted if line.strip() in ("-", "") else fg
        draw.text((margin, y), line, font=body_font, fill=color)
        y += line_h

    footer = datetime.now(ZoneInfo("Asia/Taipei")).strftime("產生時間：%Y-%m-%d %H:%M (台灣時間)")
    draw.text((margin, height - margin), footer, font=load_font(24), fill=muted)

    fmt = "JPEG" if image_format.lower() in ("jpg", "jpeg") else image_format.upper()
    save_kwargs = {"quality": 92} if fmt == "JPEG" else {}
    img.save(out_path, fmt, **save_kwargs)
    logger.info("已產生報表圖：%s (%dx%d, %s)", out_path, width, height, fmt)
    return out_path


# ── 圖片公開託管（給 IG image_url）────────────────────────────────────────

def upload_image(image_path: str | Path, *, dry_run: bool = False) -> str:
    """
    把圖片放到「可公開讀取的 HTTPS URL」，回傳該 URL（IG image_url 需要）。

    依環境變數選擇策略：
      1. IMGBB_API_KEY            -> 上傳到 imgbb，回傳直連 URL（最低成本）
      2. PUBLIC_IMAGE_BASE_URL    -> 把圖片複製到 PUBLIC_IMAGE_DIR（若有設定），
                                     URL = PUBLIC_IMAGE_BASE_URL/<檔名>（自架靜態/儀表板）
      3. 皆未設定                  -> live 模式報錯；dry-run 回傳 file:// 佔位 URL
    """
    image_path = Path(image_path)

    imgbb_key = os.getenv("IMGBB_API_KEY")
    base_url = os.getenv("PUBLIC_IMAGE_BASE_URL")
    public_dir = os.getenv("PUBLIC_IMAGE_DIR")

    if dry_run:
        if base_url:
            return f"{base_url.rstrip('/')}/{image_path.name}"
        return f"file://{image_path.resolve()}"

    if imgbb_key:
        import requests  # 延遲匯入
        with open(image_path, "rb") as fh:
            resp = requests.post(
                "https://api.imgbb.com/1/upload",
                params={"key": imgbb_key},
                files={"image": fh},
                timeout=60,
            )
        resp.raise_for_status()
        url = resp.json()["data"]["url"]
        logger.info("已上傳到 imgbb：%s", url)
        return url

    if base_url:
        if public_dir:
            import shutil
            dest = Path(public_dir)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest / image_path.name)
            logger.info("已複製圖片到公開目錄：%s", dest / image_path.name)
        return f"{base_url.rstrip('/')}/{image_path.name}"

    raise RuntimeError(
        "未設定圖片公開託管。請在 .env 設定 IMGBB_API_KEY，"
        "或 PUBLIC_IMAGE_BASE_URL(+PUBLIC_IMAGE_DIR)。"
    )


# ── Threads 發文 ──────────────────────────────────────────────────────────

def post_threads(creds: SocialCredentials, text: str, *, dry_run: bool = False, wait_seconds: float = 0.0) -> dict:
    """
    Threads 純文字貼文（兩步：建立 TEXT container -> publish）。
    host: graph.threads.net/v1.0
    """
    text = text[: THREADS_TEXT_LIMIT]
    if dry_run or not creds.threads_ready:
        if not creds.threads_ready and not dry_run:
            logger.warning("Threads 憑證未設定，改為 dry-run（不發文）。")
        print("─" * 48)
        print("[Threads] 將發布純文字貼文：")
        print(text)
        print("─" * 48)
        return {"dry_run": True, "channel": "threads", "text": text}

    import requests  # 延遲匯入
    host = os.getenv("THREADS_GRAPH_HOST", "https://graph.threads.net")
    base = f"{host}/v1.0/{creds.threads_user_id}"
    token = creds.threads_access_token

    create = requests.post(
        f"{base}/threads",
        data={"media_type": "TEXT", "text": text, "access_token": token},
        timeout=60,
    )
    create.raise_for_status()
    creation_id = create.json()["id"]

    if wait_seconds:
        time.sleep(wait_seconds)

    publish = requests.post(
        f"{base}/threads_publish",
        data={"creation_id": creation_id, "access_token": token},
        timeout=60,
    )
    publish.raise_for_status()
    media_id = publish.json().get("id")
    logger.info("Threads 發文成功，media_id=%s", media_id)
    return {"dry_run": False, "channel": "threads", "id": media_id, "creation_id": creation_id}


# ── Instagram 發文 ────────────────────────────────────────────────────────

def _wait_ig_container_ready(
    host: str,
    version: str,
    creation_id: str,
    token: str,
    *,
    initial_wait: float = 0.0,
    poll_interval: float = 3.0,
    max_polls: int = 12,
) -> None:
    """
    輪詢 IG container 的 status_code 直到 FINISHED。

    Meta 是在「建立 container」時 server-side 抓取 image_url，需要時間處理，
    發布前最好確認 FINISHED（而不是盲等）。ERROR/EXPIRED 直接報錯。
    """
    import requests  # 延遲匯入
    if initial_wait:
        time.sleep(initial_wait)
    status_url = f"{host}/{version}/{creation_id}"
    for _ in range(max_polls):
        resp = requests.get(
            status_url,
            params={"fields": "status_code,status", "access_token": token},
            timeout=60,
        )
        resp.raise_for_status()
        status = resp.json().get("status_code")
        if status == "FINISHED":
            return
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"IG container {creation_id} 狀態為 {status}：{resp.json()}")
        time.sleep(poll_interval)
    raise RuntimeError(
        f"IG container {creation_id} 在輪詢逾時前未回報 FINISHED（最後狀態={status}），中止發布。"
    )


def post_instagram(
    creds: SocialCredentials,
    image_url: str,
    caption: str,
    *,
    dry_run: bool = False,
    wait_seconds: float = 0.0,
) -> dict:
    """
    Instagram 單圖貼文（兩步：用 image_url 建立 media container -> media_publish）。
    host: graph.facebook.com/<version>（Facebook Login）。
    若用 Instagram Login，把 IG_GRAPH_HOST 設成 https://graph.instagram.com。
    """
    caption = truncate_caption(caption)
    if dry_run or not creds.instagram_ready:
        if not creds.instagram_ready and not dry_run:
            logger.warning("Instagram 憑證未設定，改為 dry-run（不發文）。")
        print("─" * 48)
        print("[Instagram] 將發布圖文：")
        print(f"image_url: {image_url}")
        print("caption:")
        print(caption)
        print("─" * 48)
        return {"dry_run": True, "channel": "instagram", "image_url": image_url, "caption": caption}

    import requests  # 延遲匯入
    host = os.getenv("IG_GRAPH_HOST", "https://graph.facebook.com")
    version = os.getenv("GRAPH_API_VERSION", "v21.0")
    base = f"{host}/{version}/{creds.ig_user_id}"
    token = creds.meta_access_token

    create = requests.post(
        f"{base}/media",
        data={"image_url": image_url, "caption": caption, "access_token": token},
        timeout=120,
    )
    create.raise_for_status()
    creation_id = create.json()["id"]

    # IG 容器需要時間處理圖片：輪詢 status_code 直到 FINISHED 再發布。
    _wait_ig_container_ready(host, version, creation_id, token, initial_wait=wait_seconds)

    publish = requests.post(
        f"{base}/media_publish",
        data={"creation_id": creation_id, "access_token": token},
        timeout=120,
    )
    publish.raise_for_status()
    media_id = publish.json().get("id")
    logger.info("Instagram 發文成功，media_id=%s", media_id)
    return {"dry_run": False, "channel": "instagram", "id": media_id, "creation_id": creation_id}


# ── 主流程 ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 DCA 中文報告自動發到 Instagram / Threads")
    parser.add_argument("--config", "-c", default="config/dca.yaml", help="DCA 設定檔（傳給 run_dca --report）")
    parser.add_argument("--post-threads", action="store_true", help="發布 Threads 純文字貼文")
    parser.add_argument("--post-instagram", action="store_true", help="發布 Instagram 圖文")
    parser.add_argument("--dry-run", action="store_true", help="只印出將發文內容，不呼叫任何 API")
    parser.add_argument("--image-out", default="reports/dca/dca_report.jpg", help="報表圖輸出路徑")
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "jpeg", "png"], help="報表圖格式")
    parser.add_argument(
        "--ig-publish-wait", type=float, default=5.0,
        help="IG 建立 container 後、發布前的等待秒數（讓 Meta 處理圖片）",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    creds = load_credentials()

    # 沒指定任何頻道時，預設兩個都做（dry-run 預覽）。
    do_threads = args.post_threads or not (args.post_threads or args.post_instagram)
    do_instagram = args.post_instagram or not (args.post_threads or args.post_instagram)

    # Threads 與 IG 通常是不同的 token；提醒 fallback 的情況（實際呼叫會 190 失敗）。
    if do_threads and not args.dry_run and creds.threads_ready and os.getenv("THREADS_ACCESS_TOKEN") is None:
        logger.warning(
            "未設定 THREADS_ACCESS_TOKEN，正在 fallback 用 META_ACCESS_TOKEN —— "
            "Threads 與 Instagram 通常是不同的 token，這多半會失敗（error 190）。"
        )

    report = get_report_text(args.config)
    print(report)  # 一律先把完整報告印到 stdout，方便排程 log 與肉眼檢查

    results: list[dict] = []

    if do_threads:
        threads_text = condense_report_for_threads(report)
        results.append(post_threads(creds, threads_text, dry_run=args.dry_run))

    if do_instagram:
        # Instagram 只吃 JPEG，PNG 會被拒；若使用者選 png 就強制改成 jpg。
        ig_format = args.image_format
        if ig_format == "png":
            logger.warning("Instagram 不支援 PNG，已自動改用 JPEG 發文。")
            ig_format = "jpg"
        image_out = args.image_out
        if ig_format in ("jpg", "jpeg") and image_out.lower().endswith(".png"):
            image_out = str(Path(image_out).with_suffix(".jpg"))
        image_path = render_report_image(report, image_out, image_format=ig_format)
        ig_dry = args.dry_run or not creds.instagram_ready
        image_url = upload_image(image_path, dry_run=ig_dry)
        results.append(
            post_instagram(
                creds,
                image_url,
                report,
                dry_run=ig_dry,
                wait_seconds=args.ig_publish_wait,
            )
        )

    posted = [r for r in results if not r.get("dry_run")]
    logger.info("完成：%d 個頻道實際發文，%d 個 dry-run。", len(posted), len(results) - len(posted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
