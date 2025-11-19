# Thu thập câu hỏi từ AI StackExchange

## 1. Giới thiệu

Dự án này tạo ra để thu thập dữ liệu câu hỏi từ mục “Newest” của trang AI StackExchange tại địa chỉ:

`https://ai.stackexchange.com/questions?tab=newest&page=...`

Script sẽ duyệt qua từng trang câu hỏi, trích xuất thông tin cơ bản của từng câu hỏi và lưu toàn bộ kết quả vào một file CSV để phục vụ phân tích sau này.

## 2. Website mục tiêu

Trang được chọn là danh sách các câu hỏi mới nhất trên AI StackExchange.

Ví dụ đường dẫn của một trang:

`https://ai.stackexchange.com/questions?tab=newest&page=1`  
`https://ai.stackexchange.com/questions?tab=newest&page=2`  

Script sẽ bắt đầu từ trang 1 và tăng dần số trang cho đến khi:

1. Server trả về mã trạng thái khác 200, hoặc  
2. Không còn khối câu hỏi nào trên trang (coi như đã hết nội dung), hoặc  
3. Đã đi đến giới hạn `max_pages`.

## 3. Dữ liệu được trích xuất

Mỗi câu hỏi sẽ được lưu thành một dòng trong file CSV `questions_with_stats.csv` với các cột sau:

| Cột    | Ý nghĩa |
|--------|--------|
| `page` | Số trang mà câu hỏi đó xuất hiện trong danh sách “Newest” |
| `title` | Tiêu đề câu hỏi |
| `link` | Đường dẫn đầy đủ đến trang chi tiết câu hỏi |
| `votes` | Số phiếu bầu (votes) của câu hỏi trên trang danh sách |
| `answers` | Số câu trả lời (answers) đã có |
| `views` | Số lượt xem (views) hiển thị trên trang danh sách |
| `tags` | Danh sách tag của câu hỏi, được nối bằng dấu phẩy |

File CSV có thể được mở bằng Excel, Google Sheets hoặc dùng trong các công cụ phân tích dữ liệu.

## 4. Công nghệ và thư viện sử dụng

Script sử dụng các thư viện Python sau:

1. `cloudscraper`  
   Dùng thay cho `requests`, hỗ trợ vượt qua một số lớp bảo vệ của Cloudflare bằng cách giả lập trình duyệt.

2. `BeautifulSoup` (từ gói `bs4`)  
   Dùng để phân tích HTML và trích xuất phần tử bằng CSS selector.

3. `csv`  
   Thư viện chuẩn của Python, dùng để ghi dữ liệu ra file CSV.

4. `time`  
   Thư viện chuẩn của Python, dùng để tạm dừng giữa các request (`sleep`) nhằm giảm tần suất truy cập.

## 5. Cài đặt môi trường

Yêu cầu có Python 3.

Cài đặt các thư viện cần thiết:

```bash
pip install cloudscraper beautifulsoup4
```

`csv` và `time` là thư viện chuẩn, không cần cài thêm.

## 6. Cách chạy script

1. Lưu đoạn code Python vào một file, ví dụ: `scrape_ai_stackexchange.py`.
2. Đảm bảo đã cài đặt đầy đủ thư viện như ở phần trên.
3. Chạy script bằng lệnh:

```bash
python scrape_ai_stackexchange.py
```

4. Sau khi chạy xong, terminal sẽ in ra tổng số câu hỏi đã thu thập và tên file CSV đã tạo, mặc định là:

`questions_with_stats.csv`

Nếu muốn thay đổi số lượng trang cần quét, chỉnh tham số:

```python
max_pages = 300
```

Ví dụ chỉ muốn quét 10 trang đầu, đặt:

```python
max_pages = 10
```

## 7. Phương pháp trích xuất dữ liệu

### 7.1. Tạo đối tượng scraper

```python
scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)
```

Đoạn code trên tạo một HTTP client giả lập trình duyệt Chrome trên Windows, giúp việc gửi request trông “giống người thật” hơn và thân thiện với các trang có bảo vệ Cloudflare cơ bản.

### 7.2. Vòng lặp phân trang

Script bắt đầu với:

```python
page = 1
max_pages = 300

while page <= max_pages:
    url = BASE_URL + str(page)
    response = scraper.get(url)
```

Ở mỗi vòng lặp, script:

1. Ghép `BASE_URL` với số trang để tạo URL hoàn chỉnh.
2. Gửi request GET đến URL đó.
3. Kiểm tra `response.status_code`:
   - Nếu khác 200, coi là bị chặn hoặc trang không tồn tại và dừng vòng lặp.
4. Dùng BeautifulSoup để phân tích HTML.

### 7.3. Xác định khối câu hỏi

```python
soup = BeautifulSoup(response.text, "html.parser")
questions = soup.select(".s-post-summary")
```

1. `BeautifulSoup` biến HTML thành cây DOM có thể duyệt.
2. CSS selector `.s-post-summary` tương ứng với mỗi khối tóm tắt câu hỏi trên trang.
3. Nếu `questions` rỗng, script kết luận là đã hết nội dung để quét và dừng.

### 7.4. Trích xuất từng trường dữ liệu

Với từng câu hỏi `q` trong `questions`, script thực hiện:

1. Tiêu đề và link

```python
title_tag = q.select_one(".s-link")
title = title_tag.get_text(strip=True)
link = "https://ai.stackexchange.com" + title_tag["href"]
```

- `.s-link` là class của thẻ chứa tiêu đề câu hỏi dạng liên kết.
- `get_text(strip=True)` lấy nội dung và loại bỏ khoảng trắng dư thừa.
- `href` là đường dẫn tương đối, nên cần nối với domain `https://ai.stackexchange.com` để thành URL đầy đủ.

2. Thống kê votes, answers, views

```python
stat_items = q.select(".s-post-summary--stats-item-number")

votes  = stat_items[0].get_text(strip=True) if len(stat_items) > 0 else "0"
answers = stat_items[1].get_text(strip=True) if len(stat_items) > 1 else "0"
views  = stat_items[2].get_text(strip=True) if len(stat_items) > 2 else "0"
```

- `.s-post-summary--stats-item-number` trả về các con số thống kê hiển thị trên card.
- Thứ tự chỉ số:
  1. `stat_items[0]` là votes  
  2. `stat_items[1]` là answers  
  3. `stat_items[2]` là views  
- Nếu thiếu chỉ số, script gán mặc định `"0"`.

3. Danh sách tag

```python
tags = [t.get_text(strip=True) for t in q.select(".post-tag")]
```

- `.post-tag` là class của các thẻ tag gắn với câu hỏi.
- Mỗi tag được lấy text, loại bỏ khoảng trắng dư, sau đó nối lại thành chuỗi bằng `", ".join(tags)` khi ghi vào CSV.

Cuối cùng, dữ liệu của một câu hỏi được thêm vào danh sách:

```python
all_data.append({
    "page": page,
    "title": title,
    "link": link,
    "votes": votes,
    "answers": answers,
    "views": views,
    "tags": ", ".join(tags)
})
```

### 7.5. Giảm tần suất request

```python
time.sleep(1.2)
page += 1
```

Sau mỗi trang, script tạm dừng 1.2 giây trước khi chuyển sang trang tiếp theo. Việc này nhằm:

1. Giảm tải cho server.
2. Hạn chế nguy cơ bị phát hiện là bot gửi request quá dày.

## 8. Ghi dữ liệu ra file CSV

Sau khi hoàn tất vòng lặp (hoặc dừng sớm vì một trong các điều kiện), script ghi toàn bộ dữ liệu thu thập được vào file:

```python
csv_file = "questions_with_stats.csv"

with open(csv_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["page", "title", "link", "votes", "answers", "views", "tags"]
    )
    writer.writeheader()
    writer.writerows(all_data)
```

- `csv.DictWriter` dùng các key trong dictionary để map thành cột.
- `writeheader()` ghi dòng tiêu đề.
- `writerows(all_data)` ghi từng dòng dữ liệu cho mỗi câu hỏi.