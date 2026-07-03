import pandas as pd
import requests
import zipfile
import os
from io import BytesIO
from datetime import date
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE   = date(2026, 2, 1)
OUTPUT_FILE  = 'zero.csv'
SYMBOL_TYPE  = 'XAUUSD_Zero_Spread' ##XAUUSD_Standart_Plus, XAUUSD_Zero_Spread, XAUUSD, XAUUSDm, XAUUSD_Raw_Spread
MAX_WORKERS  = 60        # network-bound => 64 luồng là điểm ngọt, tăng thêm không giúp nhiều
SESSION_POOL = 60        # số requests.Session tái dùng
RETRY_DELAY  = 3         # giây chờ giữa mỗi lần retry khi tải thất bại
ZIP_DIR      = 'zip'     # thư mục lưu file zip
# ─────────────────────────────────────────────────────────────────────────────

current_date = date.today()
print(f"Ngày hiện tại : {current_date}")
print(f"Luồng tải     : {MAX_WORKERS}")

# Pool session để tránh tạo TCP connection mới mỗi request
_sessions = [requests.Session() for _ in range(SESSION_POOL)]
_session_lock = threading.Lock()
_session_idx  = 0

def get_session() -> requests.Session:
    global _session_idx
    with _session_lock:
        s = _sessions[_session_idx % SESSION_POOL]
        _session_idx += 1
    return s


def normalize_to_ms(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df['time']):
        max_t = int(df['time'].max())
        if max_t > 10**14:
            df['time'] = df['time'].astype('int64') // 10**6
        elif max_t > 10**11:
            df['time'] = df['time'].astype('int64')
        else:
            df['time'] = df['time'].astype('int64') * 1000
    else:
        df['time'] = (
            pd.to_datetime(df['time'], utc=True)
            .dt.tz_localize(None)
            .astype('datetime64[ms]')
            .astype('int64')
        )
    return df


def url_exists(url: str) -> bool:
    """HEAD request để kiểm tra link có tồn tại không (HTTP 200/206)."""
    try:
        session = get_session()
        resp = session.head(url, timeout=30, allow_redirects=True)
        return resp.status_code in (200, 206)
    except Exception:
        return False


def download_zip(url: str, filepath: str) -> bool:
    """Tải 1 ZIP và lưu vào thư mục. Trả về True nếu thành công."""
    try:
        session = get_session()
        resp = session.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return False


def parse_zip(filepath: str) -> pd.DataFrame | None:
    """Đọc file ZIP từ đĩa, parse CSV bên trong, trả về DataFrame. None nếu lỗi."""
    try:
        with zipfile.ZipFile(filepath) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
        df = df.rename(columns={'Timestamp': 'time', 'Bid': 'bid', 'Ask': 'ask'})
        df = normalize_to_ms(df)
        df = df[['time', 'bid', 'ask']]
        return df
    except Exception:
        return None


def build_url_list() -> list[tuple[str, str]]:
    """Trả về list (url, label) cần tải."""
    urls = []

    # Tháng đầy đủ: start_date → tháng trước tháng hiện tại
    d = START_DATE
    end_monthly = current_date.replace(day=1)
    while d < end_monthly:
        y, m = d.year, d.month
        url = (
            f"https://ticks.ex2archive.com/ticks/{SYMBOL_TYPE}"
            f"/{y}/{m:02d}/Exness_{SYMBOL_TYPE}_{y}_{m:02d}.zip"
        )
        urls.append((url, f"{y}-{m:02d}"))
        d += relativedelta(months=1)

    y, m = current_date.year, current_date.month
    for day in range(0, current_date.day + 1):
        url = (
            f"https://ticks.ex2archive.com/ticks/{SYMBOL_TYPE}"
            f"/{y}/{m:02d}/{day:02d}/Exness_{SYMBOL_TYPE}_{y}_{m:02d}_{day:02d}.zip"
        )
        urls.append((url, f"{y}-{m:02d}-{day:02d}"))

    return urls


# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs(ZIP_DIR, exist_ok=True)
url_list = build_url_list()

pending_urls = []
valid_zips = []  # (filepath, label)

for url, label in url_list:
    filename = url.split('/')[-1]
    filepath = os.path.join(ZIP_DIR, filename)
    if os.path.exists(filepath):
        valid_zips.append((filepath, label))
    else:
        pending_urls.append((url, filepath, label))

print(f"Tổng số mốc thời gian: {len(url_list)}")
print(f"Đã có sẵn trong '{ZIP_DIR}/': {len(valid_zips)}")
print(f"Cần kiểm tra/tải: {len(pending_urls)}")

# Bước 1: HEAD-check các url chưa có
valid_urls_to_download = []
if pending_urls:
    print("\nKiểm tra link khả dụng (HEAD check)...")
    check_lock = threading.Lock()
    checked = 0

    def check_task(item):
        url, filepath, label = item
        exists = url_exists(url)
        return url, filepath, label, exists

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        check_futures = {executor.submit(check_task, item): item for item in pending_urls}
        for future in as_completed(check_futures):
            url, filepath, label, exists = future.result()
            with check_lock:
                checked += 1
                if exists:
                    valid_urls_to_download.append((url, filepath, label))
                    print(f"  [{checked:>3}/{len(pending_urls)}] EXIST {label}")
                else:
                    print(f"  [{checked:>3}/{len(pending_urls)}] NOT FOUND {label}")

    valid_urls_to_download.sort(key=lambda x: x[2]) # sort by label
    print(f"\nLink khả dụng cần tải: {len(valid_urls_to_download)}")

# Bước 2: Tải song song
if valid_urls_to_download:
    print("\nBắt đầu tải song song...")
    lock = threading.Lock()
    done = 0

    def download_job(item):
        """Tải file, retry liên tục cho đến khi thành công."""
        url, filepath, label = item
        attempt = 0
        while True:
            attempt += 1
            success = download_zip(url, filepath)
            if success:
                return filepath, label, attempt
            print(f"  [RETRY #{attempt}] {label} — tải thất bại, thử lại sau {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_job, item): item for item in valid_urls_to_download}
        for future in as_completed(futures):
            filepath, label, attempt = future.result()
            with lock:
                done += 1
                valid_zips.append((filepath, label))
                attempt_str = f" (thử lần {attempt})" if attempt > 1 else ""
                print(f"  [{done:>3}/{len(valid_urls_to_download)}] DOWNLOADED {label}{attempt_str}")
else:
    print("\nKhông có file mới cần tải.")

# Bước 3: Đọc file zip từ đĩa và hợp nhất
print(f"\nĐọc {len(valid_zips)} file ZIP và hợp nhất...")
results: dict[str, pd.DataFrame] = {}
read_lock = threading.Lock()
read_done = 0

def read_task(item):
    filepath, label = item
    df = parse_zip(filepath)
    return label, df

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(read_task, item): item for item in valid_zips}
    for future in as_completed(futures):
        label, df = future.result()
        with read_lock:
            read_done += 1
            if df is not None:
                results[label] = df
                print(f"  [{read_done:>3}/{len(valid_zips)}] READ OK  {label} ({len(df):,} ticks)")
            else:
                print(f"  [{read_done:>3}/{len(valid_zips)}] READ ERR {label}")

all_frames = [results[k] for k in sorted(results.keys())]
if all_frames:
    merged = pd.concat(all_frames, ignore_index=True)
    merged.sort_values('time', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    print(f"\nTổng ticks: {len(merged):,}")
    print(f"Ghi ra {OUTPUT_FILE}...")
    merged.to_csv(OUTPUT_FILE, index=False)
    print("Hoàn tất.")
else:
    print("\nKhông có dữ liệu để hợp nhất.")
