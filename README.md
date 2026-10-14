# IFBench tiếng Việt

Đây là bản địa hóa IFBench dành cho việc tạo và đánh giá dữ liệu tuân thủ chỉ
dẫn tiếng Việt. Mã nguồn gốc thuộc dự án **Generalizing Verifiable Instruction
Following** của Allen Institute for AI; bài báo được chấp nhận tại NeurIPS 2025.

## Những gì đã được bản địa hóa

- Toàn bộ 83 chỉ dẫn trong registry được diễn đạt tự nhiên bằng tiếng Việt.
- Từ khóa mặc định, hư từ, liên từ, đại từ, tên người, nhãn câu hỏi, mẫu đầu ra,
  tiêu đề CSV và các đáp án cố định đã được đổi sang tiếng Việt.
- Các tác vụ bảng chữ cái dùng đủ 29 chữ cái tiếng Việt.
- Bộ tách từ/câu không còn phụ thuộc vào mô hình Punkt và bộ gán nhãn từ loại
  tiếng Anh. Một “đơn vị từ” được định nghĩa rõ là một chuỗi chữ cái hoặc chữ số
  Unicode liên tiếp ở các tác vụ đếm.
- Các bộ kiểm tra có thể chấp nhận rỗng hoặc kiểm tra sai hiện vật trước đây đã
  được siết chặt, gồm ngoặc lồng, mục phụ, thụt dòng, trích dẫn và mẫu đầu ra.

Tên lớp và mã `instruction_id` cũ được giữ nguyên để tương thích API. Vì vậy,
một số tên như `english_capital` hoặc `odd_even_syllables` là tên kế thừa; mô tả
và hành vi thực tế đã được bản địa hóa.

## Cài đặt và kiểm thử

```bash
uv sync
uv run pytest -q
```

## Dùng trong mã Python

```python
from ifbench import instructions_registry

checker_cls = instructions_registry.INSTRUCTION_DICT["keywords:existence"]
checker = checker_cls("keywords:existence")
instruction = checker.build_description(keywords=["mây", "gió"])

print(instruction)
# Câu trả lời phải chứa tất cả từ khóa sau: ['gió', 'mây'].

checker.check_following("Mây trôi trong gió.")  # True
```

Khi tạo dữ liệu huấn luyện tự động, chỉ nên lấy ID từ:

```python
instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS
```

Danh sách này gồm 70 tác vụ có thể kiểm tra hoàn toàn bằng cấu trúc hoặc phép
đếm xác định. 13 tác vụ còn lại nằm trong
`VIETNAMESE_EXPERIMENTAL_REASONS`, kèm lý do không nên dùng làm nhãn thưởng tự
động tuyệt đối. Xem [VIETNAMESE_COMPATIBILITY.md](VIETNAMESE_COMPATIBILITY.md)
để biết danh sách và nguyên tắc lựa chọn.

## Lưu ý về dữ liệu đi kèm

Hai tệp `data/IFBench_test.jsonl` và `ifbench/data/IFBench_test.jsonl` là dữ liệu
kiểm thử tiếng Anh nguyên bản của upstream, được giữ lại để truy xuất nguồn gốc
và so sánh hồi quy. **Không dùng trực tiếp hai tệp này làm dữ liệu huấn luyện
tiếng Việt.** Hãy ghép các mô tả do `build_description()` sinh ra với tập câu
hỏi nền bằng tiếng Việt của bạn; mọi hiện vật phát sinh mặc định giờ đã là tiếng
Việt.

## Tạo dữ liệu train/validation cho RL

Không cần cung cấp đầu vào để tạo ngay bộ dữ liệu `core` mặc định gồm 39 tác vụ
đã qua kiểm tra thủ công, 1.000 mẫu train và 100 mẫu validation:

```bash
uv run python generate_vi_rl_dataset.py
```

Kết quả nằm trong `data/vi_rl_core/train.jsonl`, `data/vi_rl_core/val.jsonl` và
`data/vi_rl_core/manifest.json`. Mỗi mẫu có các trường IFBench (`prompt`,
`instruction_id_list`, `kwargs`) để chấm thưởng bằng bộ kiểm tra, đồng thời có
trường `messages` thuận tiện cho pipeline RL dạng hội thoại.

Có ba lựa chọn cho `--task-set`:

- `core`: 39 tác vụ có tín hiệu thưởng ổn định, dùng mặc định cho RLVR.
- `hard`: 13 tác vụ hợp lý nhưng khó hoặc có reward thưa, dùng cho curriculum.
- `all`: kết hợp 52 tác vụ của hai nhóm trên.

Số lượng train và validation được điều khiển độc lập:

```bash
uv run python generate_vi_rl_dataset.py \
  --task-set core \
  --train-size 20000 \
  --val-size 2000

uv run python generate_vi_rl_dataset.py \
  --task-set hard \
  --train-size 5000 \
  --val-size 500
```

Nếu không truyền `--output-dir`, kết quả lần lượt được ghi vào
`data/vi_rl_core`, `data/vi_rl_hard` hoặc `data/vi_rl_all`, tránh ghi đè giữa
các nhóm.

Có thể dùng ngân hàng câu hỏi riêng ở dạng TXT hoặc JSONL:

```bash
uv run python generate_vi_rl_dataset.py \
  --task-set core \
  --base-prompts du_lieu_goc.jsonl \
  --prompt-field prompt \
  --train-size 10000 \
  --val-size 1000 \
  --seed 42
```

Tệp TXT dùng một câu hỏi trên mỗi dòng. Việc chia tập và sinh tham số là xác
định theo `--seed`; generator loại câu hỏi trùng giữa train và validation.

### Gọi model và chấm thưởng song song

`generate_and_verify.py` dùng Python client chính thức của OpenAI để gọi một
API tương thích OpenAI bằng nhiều worker. Mỗi câu trả lời được chấm ngay bằng
cả bộ kiểm tra strict và loose; kết quả được ghi tăng dần nên có thể tiếp tục
sau khi gián đoạn bằng `--resume`.

```bash
export API_KEY="khóa-truy-cập-của-bạn"
uv run python generate_and_verify.py \
  --api-base http://may-chu-vllm:8000/v1 \
  --input-file data/vi_rl_core/val.jsonl \
  --workers 16 \
  --max-tokens 2048
```

Nếu server chỉ phục vụ một model, script tự lấy ID từ `/models`; nếu có nhiều
model, truyền thêm `--model`. Kết quả chi tiết và báo cáo tổng hợp được lưu mặc
định trong `results/`. Chạy thử nhanh với `--limit 10`, rồi bỏ tùy chọn này để
chạy toàn bộ dữ liệu.

## Đánh giá

Chuẩn bị một tệp JSONL đầu vào và một tệp JSONL câu trả lời theo định dạng mẫu,
sau đó chạy:

```bash
uv run python -m run_eval \
  --input_data=du_lieu_tieng_viet.jsonl \
  --input_response_data=cau_tra_loi.jsonl \
  --output_dir=eval
```

## Giấy phép và trích dẫn

Mã nguồn được phát hành theo Apache 2.0. Dữ liệu upstream dùng ODC-BY-1.0.

```bibtex
@misc{pyatkin2025generalizing,
  title={Generalizing Verifiable Instruction Following},
  author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
  year={2025},
  journal={Advances in Neural Information Processing Systems},
  volume={38}
}
```
