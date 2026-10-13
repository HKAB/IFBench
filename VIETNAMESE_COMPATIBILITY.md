# Ma trận tương thích tiếng Việt

## Tiêu chí “khuyến nghị”

Một tác vụ chỉ được đưa vào danh sách khuyến nghị khi kết quả có thể xác định từ
chuỗi đầu ra bằng quy tắc đã nêu ngay trong chỉ dẫn. Danh sách này cố ý bảo thủ:
không coi bộ nhận diện ngôn ngữ, phân tích ngữ pháp, kiến thức thực tế hoặc đánh
giá nội dung chủ đề là bằng chứng chắc chắn 100%.

Nguồn chuẩn để dùng trong mã là
`instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS`.

## 70 tác vụ khuyến nghị

### Từ khóa, phép đếm và tỉ lệ

- `keywords:existence`
- `keywords:frequency`
- `keywords:forbidden_words`
- `keywords:letter_frequency`
- `count:word_count_range`
- `count:unique_word_count`
- `count:conjunctions`
- `count:person_names`
- `count:numbers`
- `count:punctuation`
- `count:keywords_multiple`
- `ratio:sentence_type`
- `ratio:sentence_balance`
- `ratio:overlap` — chỉ dùng văn bản tham chiếu tiếng Việt
- `ratio:sentence_words`

### Độ dài và vị trí

- `length_constraints:number_sentences`
- `length_constraints:number_paragraphs`
- `length_constraints:number_words`
- `length_constraints:nth_paragraph_first_word`
- `sentence:keyword`
- `sentence:alliteration_increment`
- `sentence:increment`
- `words:keywords_specific_position`
- `words:words_position`

### Quy tắc từ và chữ cái

- `words:alphabet`
- `words:vowel`
- `words:consonants`
- `words:prime_lengths`
- `words:start_verb`
- `words:repeats`
- `words:odd_even_syllables` — ID kế thừa; bản Việt kiểm tra chẵn/lẻ số ký tự
- `words:last_first`
- `words:paragraph_last_first`
- `words:no_consecutive`

### Định dạng có thể kiểm tra chính xác

- `detectable_content:number_placeholders`
- `detectable_content:postscript`
- `detectable_format:number_bullet_lists`
- `detectable_format:constrained_response`
- `detectable_format:number_highlighted_sections`
- `detectable_format:multiple_sections`
- `detectable_format:json_format`
- `detectable_format:title`
- `format:parentheses`
- `format:quotes`
- `format:options`
- `format:newline`
- `format:line_indent`
- `format:quote_unquote`
- `format:list`
- `format:thesis`
- `format:sub-bullets`
- `format:no_bullets_bullets`
- `format:title_case`
- `format:output_template`
- `format:no_whitespace`
- `punctuation:no_comma`
- `startend:quotation`
- `startend:end_checker`
- `change_case:capital_word_frequency`

### Kết hợp, lặp lại và tác vụ cố định

- `combination:two_responses`
- `combination:repeat_prompt`
- `repeat:repeat_change`
- `repeat:repeat_simple`
- `repeat:repeat_span`
- `custom:multiples`
- `custom:reverse_newline`
- `custom:sentence_alphabet`
- `custom:csv_city`
- `custom:csv_special_character`
- `custom:csv_quotes`

## 13 tác vụ thử nghiệm — không dùng làm nhãn tự động tuyệt đối

| ID | Lý do |
|---|---|
| `language:response_language` | `langdetect` chỉ ước lượng ngôn ngữ, đặc biệt không ổn định với câu ngắn. |
| `change_case:english_capital` | Phần chữ hoa chính xác nhưng kết quả cuối còn phụ thuộc `langdetect`; tên ID là tên kế thừa. |
| `change_case:english_lowercase` | Phần chữ thường chính xác nhưng kết quả cuối còn phụ thuộc `langdetect`; tên ID là tên kế thừa. |
| `ratio:stop_words` | Tiếng Việt không có một danh sách hư từ chuẩn duy nhất. |
| `words:palindrome` | Khó tạo đủ từ đơn tiếng Việt tự nhiên dài ít nhất 5 ký tự. |
| `format:emoji` | Không có quy ước duy nhất cho ranh giới câu khi emoji đứng trước hoặc sau dấu câu. |
| `count:words_japanese` | Đây là tác vụ trộn tiếng Nhật, không phải đầu ra thuần tiếng Việt. |
| `count:pronouns` | Tư cách đại từ và ranh giới cụm đại từ tiếng Việt phụ thuộc ngữ cảnh. |
| `custom:mcq_count_length` | Bộ kiểm tra đếm cấu trúc nhưng không xác minh chủ đề hoặc tính đúng đắn của câu hỏi. |
| `custom:word_reverse` | Bộ kiểm tra tìm đáp án con, không chứng minh toàn bộ câu đảo là hợp lệ. |
| `custom:character_reverse` | Bộ kiểm tra tìm đáp án con, không chứng minh toàn bộ chuỗi đảo là hợp lệ. |
| `custom:european_capitals_sort` | Phụ thuộc dữ liệu địa chính trị, tọa độ và quy ước gọi địa danh. |
| `custom:date_format_list` | Bộ kiểm tra định dạng ngày không thể xác minh danh sách đầy đủ mọi trận chiến. |

Các tác vụ thử nghiệm vẫn được dịch và có bộ kiểm tra hoạt động để phục vụ nghiên
cứu, nhưng cần duyệt tay hoặc một bộ xác minh ngữ nghĩa bổ sung trước khi đưa vào
dữ liệu huấn luyện.
