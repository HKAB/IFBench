#!/usr/bin/env python3
"""Generate deterministic Vietnamese IFBench train/validation JSONL files.

The default configuration needs no input file. It combines a small bundled
bank of natural Vietnamese requests with the 39 categories that passed the
manual 31B audit. A separate hard tier contains 13 logically valid but sparse
or difficult categories. A custom TXT or JSONL prompt bank can be supplied
with ``--base-prompts``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ifbench import instructions_registry
from ifbench.instructions_util import VIETNAMESE_WORD_LIST


# Nhóm cốt lõi đã được duyệt thủ công trên hai mẫu cho mỗi tác vụ. Các prompt
# rõ nghĩa, checker khớp với cách đọc tự nhiên và model 31B tạo được tín hiệu
# thưởng đủ dùng cho giai đoạn RLVR đầu tiên.
CORE_INSTRUCTION_IDS = frozenset({
    "change_case:capital_word_frequency",
    "combination:repeat_prompt",
    "combination:two_responses",
    "count:conjunctions",
    "count:keywords_multiple",
    "count:numbers",
    "count:punctuation",
    "count:unique_word_count",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:title",
    "format:line_indent",
    "format:list",
    "format:no_bullets_bullets",
    "format:no_whitespace",
    "format:output_template",
    "format:parentheses",
    "format:quote_unquote",
    "format:sub-bullets",
    "format:title_case",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "ratio:sentence_balance",
    "repeat:repeat_change",
    "sentence:alliteration_increment",
    "sentence:keyword",
    "startend:end_checker",
    "startend:quotation",
    "words:last_first",
    "words:start_verb",
})

# Nhóm khó vẫn có prompt/checker hợp lý nhưng reward rất thưa với model 31B.
# Nên dùng ở giai đoạn curriculum sau khi model đã học tốt nhóm core.
HARD_INSTRUCTION_IDS = frozenset({
    "count:word_count_range",
    "custom:sentence_alphabet",
    "format:quotes",
    "format:thesis",
    "ratio:overlap",
    "ratio:sentence_type",
    "ratio:sentence_words",
    "repeat:repeat_span",
    "sentence:increment",
    "words:keywords_specific_position",
    "words:no_consecutive",
    "words:paragraph_last_first",
    "words:words_position",
})

TASK_SETS = {
    "core": CORE_INSTRUCTION_IDS,
    "hard": HARD_INSTRUCTION_IDS,
    "all": CORE_INSTRUCTION_IDS | HARD_INSTRUCTION_IDS,
}

assert len(CORE_INSTRUCTION_IDS) == 39
assert len(HARD_INSTRUCTION_IDS) == 13
assert CORE_INSTRUCTION_IDS.isdisjoint(HARD_INSTRUCTION_IDS)
assert (CORE_INSTRUCTION_IDS | HARD_INSTRUCTION_IDS).issubset(
    instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS)


DEFAULT_BASE_PROMPTS = (
    "Giải thích vì sao cây xanh cần ánh sáng để phát triển.",
    "Nêu ba cách tiết kiệm điện trong gia đình.",
    "So sánh ưu điểm của xe buýt và xe đạp khi đi lại trong thành phố.",
    "Viết một đoạn giới thiệu ngắn về văn hóa uống trà ở Việt Nam.",
    "Đề xuất kế hoạch học từ vựng trong bốn tuần cho người mới bắt đầu.",
    "Giải thích hiện tượng cầu vồng bằng ngôn ngữ dễ hiểu.",
    "Trình bày lợi ích của việc đọc sách mỗi ngày.",
    "Gợi ý một thực đơn lành mạnh cho bữa sáng.",
    "Mô tả khung cảnh một khu chợ quê vào buổi sớm.",
    "Nêu các bước chuẩn bị cho một buổi thuyết trình hiệu quả.",
    "Giải thích sự khác nhau giữa thời tiết và khí hậu.",
    "Đề xuất cách tổ chức góc học tập gọn gàng và yên tĩnh.",
    "Viết lời khuyên giúp một người hình thành thói quen tập thể dục.",
    "Tóm tắt vai trò của rừng ngập mặn đối với môi trường ven biển.",
    "Lập kế hoạch cho chuyến tham quan bảo tàng kéo dài nửa ngày.",
    "Mô tả cách nấu một món canh rau đơn giản.",
    "Giải thích vì sao cần kiểm chứng nguồn tin trước khi chia sẻ.",
    "Nêu một số biện pháp giảm rác thải nhựa ở trường học.",
    "Viết đoạn văn khuyến khích mọi người sử dụng thư viện công cộng.",
    "Đề xuất hoạt động gắn kết cho một nhóm làm việc từ xa.",
    "Giải thích cách lập ngân sách cá nhân cơ bản cho sinh viên.",
    "Mô tả vòng tuần hoàn của nước trong tự nhiên.",
    "So sánh việc học trực tiếp và học trực tuyến.",
    "Gợi ý cách chăm sóc một cây non trong mùa khô.",
    "Viết một câu chuyện ngắn về tình bạn và lòng trung thực.",
    "Nêu các nguyên tắc ứng xử lịch sự trong cuộc họp.",
    "Giải thích cách hoạt động của pin mặt trời ở mức phổ thông.",
    "Đề xuất lịch trình khám phá Huế trong một ngày.",
    "Trình bày ý nghĩa của việc bảo tồn các làng nghề truyền thống.",
    "Gợi ý phương pháp ghi chép giúp nhớ bài lâu hơn.",
    "Mô tả một buổi chiều mưa ở Hà Nội.",
    "Giải thích lợi ích của việc ngủ đủ giấc.",
    "Nêu cách xử lý khi hai thành viên trong nhóm bất đồng ý kiến.",
    "Đề xuất ba hoạt động khoa học đơn giản cho trẻ em tại nhà.",
    "Viết phần giới thiệu cho một câu lạc bộ nhiếp ảnh.",
    "Giải thích vì sao đa dạng sinh học quan trọng.",
    "Lập danh sách việc cần làm trước một chuyến đi xa.",
    "Mô tả cách pha một cốc nước chanh ít đường.",
    "Nêu lợi ích và hạn chế của làm việc tại nhà.",
    "Đề xuất cách giúp khu phố có thêm không gian xanh.",
    "Giải thích nguyên nhân thủy triều lên xuống bằng ngôn ngữ phổ thông.",
    "Nêu những lưu ý để đi xe máy an toàn khi trời mưa.",
    "Soạn lời từ chối một lời mời dự sự kiện sao cho lịch sự.",
    "Phân tích lợi ích và rủi ro của trí tuệ nhân tạo trong giáo dục.",
    "Hướng dẫn làm món cơm rang từ những nguyên liệu còn lại trong tủ lạnh.",
    "Mô tả không khí ngày Tết ở một gia đình Việt Nam.",
    "Đề xuất cách phân chia công việc công bằng khi làm bài tập nhóm.",
    "Giải thích vì sao bầu trời có màu xanh vào ban ngày.",
    "Viết đoạn giới thiệu hấp dẫn cho một lễ hội sách của trường.",
    "Nêu cách sơ cứu cơ bản khi bị bỏng nhẹ trong sinh hoạt.",
    "Lập kế hoạch ôn thi trong bảy ngày mà vẫn có thời gian nghỉ ngơi.",
    "So sánh năng lượng gió và năng lượng mặt trời.",
    "Gợi ý cách trò chuyện với trẻ em về việc bảo vệ động vật hoang dã.",
    "Mô tả hành trình của một hạt cà phê từ nông trại đến tách cà phê.",
    "Giải thích cách nhận biết một mật khẩu mạnh và an toàn.",
    "Đề xuất hoạt động cuối tuần phù hợp cho gia đình có trẻ nhỏ.",
    "Viết lời cảm ơn dành cho một người đã giúp đỡ mình lúc khó khăn.",
    "Trình bày các yếu tố tạo nên một cuộc phỏng vấn xin việc hiệu quả.",
    "Giải thích vì sao âm thanh không truyền được trong chân không.",
    "Gợi ý cách tận dụng ban công nhỏ để trồng rau gia vị.",
    "Nêu phương án xử lý khi bị lạc đường ở một thành phố xa lạ.",
    "Mô tả nét đặc trưng của kiến trúc nhà sàn ở Việt Nam.",
    "Đề xuất cách duy trì sự tập trung khi học trong môi trường ồn ào.",
    "Giải thích tác dụng của việc phân loại rác tại nguồn.",
    "Viết một đoạn hội thoại hòa giải giữa hai người bạn đang hiểu lầm nhau.",
    "Lập kế hoạch tổ chức buổi sinh hoạt câu lạc bộ với ngân sách hạn chế.",
    "So sánh sách giấy và sách điện tử từ góc nhìn của người đọc.",
    "Nêu những việc cần kiểm tra trước khi mua một thiết bị điện tử cũ.",
    "Giải thích quá trình lên men trong việc làm sữa chua.",
    "Gợi ý các cách giữ gìn sức khỏe tinh thần trong giai đoạn bận rộn.",
    "Mô tả một đêm ngắm sao ở vùng nông thôn.",
    "Viết lời quảng bá ngắn cho một sản phẩm thủ công thân thiện với môi trường.",
    "Đề xuất cách cải thiện trải nghiệm của hành khách tại bến xe.",
    "Giải thích vì sao cần cập nhật phần mềm thường xuyên.",
    "Nêu cách chuẩn bị cho một cuộc trao đổi khó khăn với đồng nghiệp.",
    "Trình bày vai trò của âm nhạc trong đời sống cộng đồng.",
    "Gợi ý một thí nghiệm đơn giản để minh họa áp suất không khí.",
    "Mô tả quy trình mượn và trả sách tại một thư viện hiện đại.",
    "Giải thích cách đọc nhãn dinh dưỡng trên bao bì thực phẩm.",
    "Đề xuất giải pháp giảm ùn tắc giao thông quanh cổng trường.",
    "Viết đoạn kết cho câu chuyện về một người tìm lại món đồ kỷ niệm.",
    "Nêu những nguyên tắc bảo vệ quyền riêng tư khi dùng mạng xã hội.",
    "So sánh du lịch tự túc và du lịch theo đoàn.",
    "Giải thích vì sao vận động nhẹ sau khi ngồi lâu có lợi cho cơ thể.",
    "Lập kế hoạch gây quỹ cho một hoạt động thiện nguyện tại địa phương.",
    "Mô tả vẻ đẹp của ruộng bậc thang vào mùa lúa chín.",
    "Gợi ý cách phản hồi mang tính xây dựng cho bài viết của bạn học.",
    "Trình bày các bước lựa chọn nguồn tài liệu đáng tin cậy cho bài nghiên cứu.",
    "Giải thích nguyên lý hoạt động cơ bản của hệ thống định vị toàn cầu.",
    "Đề xuất cách thiết kế một ngày không sử dụng đồ nhựa dùng một lần.",
    "Viết thông báo ngắn về việc thay đổi thời gian của một cuộc họp.",
    "Nêu cách bảo quản rau củ để hạn chế lãng phí thực phẩm.",
    "Mô tả cảm giác lần đầu học được một kỹ năng mới.",
    "Giải thích sự khác nhau giữa dữ liệu và thông tin bằng ví dụ dễ hiểu.",
    "Gợi ý cách chào đón một thành viên mới vào nhóm làm việc.",
    "Phân tích ảnh hưởng của ánh sáng nhân tạo đến giấc ngủ.",
    "Lập lịch chăm sóc thú cưng trong một tuần cho người bận rộn.",
    "Viết lời dẫn cho một chương trình phát thanh về ẩm thực vùng miền.",
    "Nêu các bước ứng phó an toàn khi xảy ra mất điện kéo dài.",
    "Đề xuất một dự án nhỏ giúp học sinh tìm hiểu lịch sử địa phương.",
    # --- SCIENCE & NATURE (41-70) ---
    "Giải thích cơ chế hoạt động của vaccine trong cơ thể.",
    "So sánh sự khác nhau giữa tế bào thực vật và tế bào động vật.",
    "Tại sao nước biển lại mặn?",
    "Giải thích hiện tượng nhật thực và nguyệt thực.",
    "Mô tả quá trình quang hợp của cây xanh.",
    "Tại sao chúng ta lại nhìn thấy sấm sét sau khi nghe thấy tiếng sấm?",
    "Giải thích nguyên lý hoạt động của nam châm.",
    "Tại sao các hành tinh trong hệ mặt trời lại quay quanh mặt trời?",
    "Phân tích tầm quan trọng của tầng ozone đối với Trái đất.",
    "Giải thích hiện tượng thủy triều.",
    "Tại sao kim loại lại dẫn điện tốt hơn nhựa?",
    "Mô tả vòng đời của một con bướm.",
    "Giải thích cách loài chim di cư tìm được đường đi.",
    "Tại sao băng lại nổi trên mặt nước?",
    "Giải thích tác động của hiệu ứng nhà kính.",
    "So sánh đặc điểm của động vật có vú và bò sát.",
    "Tại sao chúng ta cảm thấy lạnh khi gió thổi?",
    "Giải thích cách tai người tiếp nhận âm thanh.",
    "Tại sao dầu không tan trong nước?",
    "Giải thích hiện tượng khúc xạ ánh sáng khi cắm ống hút vào ly nước.",
    "Mô tả vai trò của vi khuẩn trong phân hủy chất hữu cơ.",
    "Tại sao lá cây chuyển màu vào mùa thu?",
    "Giải thích sự khác biệt giữa axit và bazơ.",
    "Tại sao áp suất khí quyển giảm khi lên cao?",
    "Giải thích cơ chế của phản xạ không điều kiện.",
    "Tại sao muối lại làm tan băng trên đường?",
    "Mô tả cách hoạt động của một chiếc kính hiển vi.",
    "Giải thích hiện tượng nóng lên toàn cầu.",
    "Tại sao một số loài vật lại ngủ đông?",
    "Giải thích nguyên lý cân bằng của con quay.",

    # --- LIFE SKILLS & PERSONAL DEVELOPMENT (71-100) ---
    "Đề xuất phương pháp quản lý thời gian Pomodoro cho học sinh.",
    "Viết hướng dẫn cách từ chối một lời mời lịch sự nhưng quyết liệt.",
    "Gợi ý các bước để vượt qua nỗi sợ nói trước đám đông.",
    "Lập danh sách các kỹ năng mềm cần thiết cho sinh viên mới tốt nghiệp.",
    "Đề xuất cách rèn luyện tư duy phản biện trong cuộc sống hàng ngày.",
    "Viết một bài hướng dẫn cách đọc nhanh mà vẫn hiểu sâu.",
    "Gợi ý cách đối phó với áp lực thi cử cho học sinh cuối cấp.",
    "Nêu các bước để giải quyết xung đột trong tình bạn.",
    "Đề xuất cách xây dựng lòng tự tin cho người hướng nội.",
    "Viết một bản kế hoạch đặt mục tiêu theo mô hình SMART.",
    "Gợi ý cách duy trì động lực khi học một ngôn ngữ mới.",
    "Hướng dẫn cách lập danh sách ưu tiên công việc (To-do list) hiệu quả.",
    "Nêu cách rèn luyện sự kiên nhẫn trong môi trường làm việc áp lực.",
    "Đề xuất cách cải thiện kỹ năng lắng nghe chủ động.",
    "Viết lời khuyên để cân bằng giữa công việc và cuộc sống.",
    "Gợi ý cách viết một email xin việc chuyên nghiệp.",
    "Nêu các quy tắc an toàn cơ bản khi sử dụng internet.",
    "Đề xuất cách rèn luyện thói quen dậy sớm.",
    "Hướng dẫn cách thực hiện thiền định cơ bản cho người mới.",
    "Viết cách quản lý cảm xúc khi gặp chuyện tiêu cực.",
    "Gợi ý cách tạo lập thói quen đọc sách 15 phút mỗi ngày.",
    "Nêu cách tự học một kỹ năng mới thông qua internet.",
    "Đề xuất cách sắp xếp tài chính cho một chuyến du lịch tự túc.",
    "Viết hướng dẫn cách ghi chép bài giảng bằng sơ đồ tư duy (Mindmap).",
    "Gợi ý cách xây dựng mối quan hệ tốt với đồng nghiệp.",
    "Nêu tầm quan trọng của việc xin lỗi chân thành.",
    "Đề xuất cách rèn luyện tư duy sáng tạo thông qua vẽ hoặc viết.",
    "Hướng dẫn cách chọn lọc thông tin giữa biển dữ liệu số.",
    "Viết lời khuyên về việc duy trì sự tò mò và tinh thần học hỏi.",
    "Gợi ý cách vượt qua cảm giác trì hoãn (procrastination).",

    # --- CULTURE, ARTS & SOCIETY (101-130) ---
    "Viết một đoạn văn miêu tả vẻ đẹp của tà áo dài Việt Nam.",
    "Giải thích ý nghĩa của ngày Tết Nguyên Đán đối với người Việt.",
    "So sánh đặc điểm của nhạc dân gian và nhạc hiện đại.",
    "Viết giới thiệu về một danh lam thắng cảnh nổi tiếng tại miền Trung.",
    "Đề xuất cách quảng bá du lịch địa phương thông qua mạng xã hội.",
    "Phân tích giá trị của sự hiếu thảo trong văn hóa Á Đông.",
    "Mô tả một lễ hội truyền thống mà bạn yêu thích.",
    "Viết đoạn văn bàn về sự thay đổi của gia đình truyền thống sang gia đình hiện đại.",
    "Giải thích vai trò của bảo tàng trong việc lưu giữ lịch sử.",
    "Gợi ý các hoạt động để giới thiệu văn hóa Việt Nam cho người nước ngoài.",
    "Viết cảm nhận về một bài thơ hoặc tác phẩm văn học kinh điển.",
    "So sánh phong tục đón năm mới ở Việt Nam và một quốc gia khác.",
    "Mô tả ý nghĩa của các biểu tượng trong kiến trúc đình chùa Việt Nam.",
    "Viết đoạn văn về tầm quan trọng của việc giữ gìn tiếng mẹ đẻ.",
    "Đề xuất cách tổ chức một triển lãm nghệ thuật nhỏ tại trường học.",
    "Giải thích sự ảnh hưởng của internet đến cách giao tiếp của giới trẻ.",
    "Viết về vai trò của người phụ nữ trong xã hội hiện đại.",
    "Mô tả sự khác biệt giữa lối sống thành thị và nông thôn.",
    "Phân tích tác động của phim ảnh đến nhận thức cộng đồng.",
    "Viết một đoạn giới thiệu về ẩm thực đường phố Sài Gòn.",
    "Giải thích ý nghĩa của tinh thần tương thân tương ái.",
    "Đề xuất cách bảo tồn các bài hát ru truyền thống.",
    "Viết về tầm quan trọng của việc tôn trọng sự khác biệt văn hóa.",
    "Mô tả một món ăn truyền thống đặc trưng của vùng quê bạn.",
    "Phân tích lợi ích của việc học nghệ thuật đối với trẻ em.",
    "Viết về sự phát triển của văn hóa đọc trong thời đại số.",
    "Giải thích khái niệm 'văn hóa doanh nghiệp'.",
    "Đề xuất cách tổ chức ngày hội văn hóa tại một khu dân cư.",
    "Viết về ý nghĩa của các câu ca dao, tục ngữ trong giáo dục đạo đức.",
    "Mô tả sự vận động của thời trang qua các thập kỷ.",

    # --- TECHNOLOGY & DIGITAL WORLD (131-160) ---
    "Giải thích khái niệm Trí tuệ nhân tạo (AI) cho một đứa trẻ.",
    "So sánh ưu và nhược điểm của điện thoại thông minh so với máy tính bảng.",
    "Viết về tác động của mạng xã hội đối với sức khỏe tâm thần.",
    "Giải thích cách hoạt động của công nghệ Blockchain một cách đơn giản.",
    "Đề xuất các biện pháp bảo mật mật khẩu cá nhân an toàn.",
    "Viết về tiềm năng của xe điện trong việc giảm ô nhiễm môi trường.",
    "Giải thích khái niệm 'Điện toán đám mây' (Cloud Computing).",
    "So sánh trải nghiệm mua sắm trực tuyến và mua sắm truyền thống.",
    "Nêu các rủi ro khi chia sẻ quá nhiều thông tin cá nhân lên mạng.",
    "Viết về tương lai của giáo dục khi có sự hỗ trợ của VR/AR.",
    "Giải thích cách một công cụ tìm kiếm (như Google) hoạt động.",
    "Đề xuất cách sử dụng công nghệ để hỗ trợ người khuyết tật.",
    "Viết về sự phát triển của thương mại điện tử tại Việt Nam.",
    "Giải thích hiện tượng 'tin giả' (fake news) lan truyền nhanh như thế nào.",
    "So sánh hệ điều hành Android và iOS.",
    "Viết về vai trò của tự động hóa trong sản xuất công nghiệp.",
    "Giải thích khái niệm Internet of Things (IoT) và ví dụ thực tế.",
    "Đề xuất cách giảm thời gian sử dụng màn hình (screen time) cho trẻ em.",
    "Viết về tác động của tiền điện tử đối với hệ thống tài chính.",
    "Giải thích cách hoạt động của định vị toàn cầu (GPS).",
    "So sánh việc lưu trữ dữ liệu vật lý và lưu trữ số.",
    "Viết về tầm quan trọng của việc sao lưu dữ liệu thường xuyên.",
    "Giải thích cơ chế của xác thực hai yếu tố (2FA).",
    "Đề xuất cách ứng dụng AI trong việc học ngoại ngữ.",
    "Viết về sự tiến hóa của các thiết bị lưu trữ (từ đĩa mềm đến SSD).",
    "Giải thích khái niệm mã nguồn mở (Open Source).",
    "Viết về ảnh hưởng của game online đến tư duy chiến thuật.",
    "So sánh làm việc trực tiếp và làm việc thông qua các công cụ quản lý task.",
    "Giải thích cách hoạt động của sóng Wi-Fi.",
    "Viết về vai trò của an ninh mạng trong quốc phòng.",

    # --- ENVIRONMENT & SUSTAINABILITY (161-180) ---
    "Đề xuất 5 cách giảm thiểu rác thải thực phẩm trong gia đình.",
    "Giải thích tại sao chúng ta nên hạn chế sử dụng túi nilon.",
    "Viết một thông điệp kêu gọi mọi người bảo vệ nguồn nước sạch.",
    "Giải thích khái niệm 'Kinh tế tuần hoàn'.",
    "Đề xuất phương pháp trồng rau sạch tại ban công căn hộ.",
    "Viết về tác hại của ô nhiễm tiếng ồn trong đô thị.",
    "Giải thích tầm quan trọng của việc phân loại rác tại nguồn.",
    "Đề xuất cách tổ chức một chiến dịch dọn rác tại bãi biển.",
    "Viết về ảnh hưởng của việc phá rừng đến biến đổi khí hậu.",
    "Giải thích tại sao cần bảo vệ các loài động vật quý hiếm.",
    "So sánh tác động của xe xăng và xe điện đến môi trường.",
    "Đề xuất cách thiết kế một ngôi nhà tiết kiệm năng lượng.",
    "Viết về vai trò của các khu bảo tồn thiên nhiên.",
    "Giải thích hiện tượng nóng lên của đại dương.",
    "Đề xuất các sản phẩm thay thế nhựa dùng một lần.",
    "Viết về lợi ích của việc sử dụng phương tiện công cộng.",
    "Giải thích cơ chế của phân bón hữu cơ đối với đất.",
    "Đề xuất cách giáo dục trẻ em về ý thức bảo vệ môi trường.",
    "Viết về sự nguy hiểm của rác thải điện tử.",

    # --- CREATIVE & IMAGINATIVE (181-200) ---
    "Viết một đoạn mô tả thế giới vào năm 2100 theo tưởng tượng của bạn.",
    "Sáng tác một bài thơ ngắn về mùa thu Hà Nội.",
    "Viết một đoạn hội thoại giữa một con mèo và một con chó về tình bạn.",
    "Mô tả một hành tinh giả tưởng nơi mọi thứ đều làm bằng kẹo.",
    "Viết một bức thư gửi cho chính bản thân mình 10 năm sau.",
    "Sáng tạo một công thức món ăn mới kết hợp giữa Á và Âu.",
    "Viết một kịch bản ngắn về một cuộc gặp gỡ bất ngờ trên tàu hỏa.",
    "Mô tả cảm giác của một hạt mưa khi rơi từ trên trời xuống đất.",
    "Viết một câu chuyện ngắn về một chiếc đồng hồ có thể dừng thời gian.",
    "Sáng tác một lời bài hát về tình yêu quê hương.",
    "Mô tả một thành phố dưới đáy đại dương.",
    "Viết một đoạn văn biện luận: 'Nếu con người có thể bay, thế giới sẽ thay đổi ra sao?'.",
    "Sáng tạo một nhân vật siêu anh hùng bảo vệ môi trường và mô tả sức mạnh của họ.",
    "Viết một đoạn giới thiệu cho một cuốn tiểu thuyết trinh thám.",
    "Mô tả một khu vườn bí mật chứa những loài hoa biết nói.",
    "Viết một bức thư từ một phi hành gia đang ở trên sao Hỏa gửi về Trái đất.",
    "Sáng tác một câu chuyện ngụ ngôn về sự kiêu ngạo.",
    "Mô tả một ngày trong cuộc đời của một bức tượng trong bảo tàng.",
    "Viết một đoạn văn về một giấc mơ kỳ lạ mà bạn từng gặp.",
    "Sáng tạo một luật chơi cho một trò chơi dân gian mới.",
)

# Các nhiệm vụ này đã tự chứa nội dung cần trả lời. Thêm một câu hỏi nền sẽ
# tạo hai yêu cầu cạnh tranh nhau, nên chỉ dùng lời dẫn trung tính.
SELF_CONTAINED_IDS = frozenset({
    "custom:multiples",
    "custom:reverse_newline",
    "custom:sentence_alphabet",
    "custom:csv_city",
    "custom:csv_special_character",
    "custom:csv_quotes",
    "detectable_format:constrained_response",
    "format:options",
    "repeat:repeat_simple",
})

FRAMES = (
    "Hãy thực hiện yêu cầu sau.",
    "Vui lòng làm đúng chỉ dẫn dưới đây.",
    "Thực hiện chính xác nhiệm vụ sau.",
    "Hãy trả lời theo yêu cầu được nêu bên dưới.",
)


def _load_prompts(path: Path | None, prompt_field: str) -> list[str]:
    if path is None:
        return list(DEFAULT_BASE_PROMPTS)
    if not path.is_file():
        raise ValueError(f"Không tìm thấy tệp câu hỏi nền: {path}")

    prompts: list[str] = []
    if path.suffix.casefold() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if prompt_field not in value or not isinstance(value[prompt_field], str):
                raise ValueError(
                    f"Dòng {line_number} không có trường chuỗi {prompt_field!r}.")
            prompts.append(value[prompt_field].strip())
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise ValueError("Tệp câu hỏi nền không có câu hỏi hợp lệ.")
    if any(len(prompt.split()) < 2 for prompt in prompts):
        raise ValueError("Mỗi câu hỏi nền phải có ít nhất hai từ.")
    return prompts


def _sample_words(rng: random.Random, count: int) -> list[str]:
    return rng.sample(VIETNAMESE_WORD_LIST, count)


def _build_kwargs(instruction_id: str, base_prompt: str,
                  rng: random.Random) -> dict[str, Any]:
    """Use moderate, learnable values instead of several very large defaults."""
    words = _sample_words(rng, 6)
    factories: dict[str, dict[str, Any]] = {
        "keywords:existence": {"keywords": words[:2]},
        "keywords:frequency": {
            "keyword": words[0], "frequency": rng.randint(1, 3),
            "relation": "at least",
        },
        "keywords:forbidden_words": {"forbidden_words": words[:2]},
        "keywords:letter_frequency": {
            "letter": rng.choice("aemnothiu"),
            "let_frequency": rng.randint(2, 5),
            "let_relation": "at least",
        },
        "length_constraints:number_sentences": {
            "num_sentences": rng.randint(2, 6), "relation": "at least",
        },
        "length_constraints:number_paragraphs": {
            "num_paragraphs": rng.randint(2, 4),
        },
        "length_constraints:number_words": {
            "num_words": rng.randint(40, 90), "relation": "at least",
        },
        "length_constraints:nth_paragraph_first_word": {
            "num_paragraphs": 3, "nth_paragraph": rng.randint(1, 3),
            "first_word": words[0],
        },
        "detectable_content:number_placeholders": {
            "num_placeholders": rng.randint(1, 3),
        },
        "detectable_content:postscript": {
            "postscript_marker": rng.choice(("Tái bút:", "Ghi chú cuối:", "P.S.:")),
        },
        "detectable_format:number_bullet_lists": {"num_bullets": rng.randint(2, 5)},
        "detectable_format:number_highlighted_sections": {
            "num_highlights": rng.randint(1, 3),
        },
        "detectable_format:multiple_sections": {
            "section_spliter": "Phần", "num_sections": rng.randint(2, 4),
        },
        "combination:repeat_prompt": {"prompt_to_repeat": base_prompt},
        "startend:end_checker": {
            "end_phrase": rng.choice(("Hết.", "Xin cảm ơn.", "Kết thúc.")),
        },
        "change_case:capital_word_frequency": {
            "capital_frequency": rng.randint(2, 5), "capital_relation": "at least",
        },
        "count:word_count_range": {
            "min_words": (minimum := rng.randint(40, 80)),
            "max_words": minimum + rng.randint(10, 25),
        },
        "count:unique_word_count": {"N": rng.randint(25, 50)},
        "count:conjunctions": {"small_n": rng.randint(2, 4)},
        "count:person_names": {"N": rng.randint(2, 5)},
        "ratio:overlap": {
            "reference_text": base_prompt, "percentage": rng.choice((25, 30, 35)),
        },
        "count:numbers": {"N": rng.randint(2, 5)},
        "format:options": {"options": "a), b), c), d)"},
        "words:repeats": {"small_n": rng.randint(3, 5)},
        "sentence:keyword": {"word": words[0], "N": rng.randint(2, 4)},
        "sentence:increment": {"small_n": rng.randint(2, 4)},
        "format:list": {"sep": rng.choice(("MỤC", "PHẦN", "Ý"))},
        "words:keywords_specific_position": {
            "keyword": words[0], "n": rng.randint(2, 4), "m": rng.randint(3, 8),
        },
        "words:words_position": {"keyword": words[0]},
        "repeat:repeat_change": {"prompt_to_repeat": base_prompt},
        "repeat:repeat_span": {
            "prompt_to_repeat": base_prompt,
            "n_start": (start := rng.randint(0, len(base_prompt.split()) - 2)),
            "n_end": rng.randint(start + 1, len(base_prompt.split()) - 1),
        },
    }
    return factories.get(instruction_id, {})


def _make_example(instruction_id: str, base_prompt: str, key: int,
                  rng: random.Random, task_set: str) -> dict[str, Any]:
    checker_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    checker = checker_cls(instruction_id)
    description = checker.build_description(
        **_build_kwargs(instruction_id, base_prompt, rng))
    kwargs = checker.get_instruction_args() or {}

    if instruction_id in SELF_CONTAINED_IDS:
        prompt = f"{rng.choice(FRAMES)}\n\n{description}"
    elif instruction_id == "repeat:repeat_change":
        prompt = description
    elif instruction_id == "ratio:overlap":
        prompt = f"{rng.choice(FRAMES)}\n\n{description}"
    elif instruction_id == "repeat:repeat_span":
        prompt = f"Văn bản nguồn: {base_prompt}\n\n{description}"
    else:
        prompt = f"{base_prompt}\n\nYêu cầu bắt buộc:\n{description}"

    return {
        "key": key,
        "task_set": task_set,
        "instruction_id_list": [instruction_id],
        "prompt": prompt,
        "kwargs": [kwargs],
        "messages": [{"role": "user", "content": prompt}],
        "verifier": {
            "type": "ifbench",
            "instruction_id_list": [instruction_id],
            "kwargs": [kwargs],
        },
    }


def _generate_examples(count: int, prompts: list[str], seed: int,
                       key_offset: int, task_set: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    instruction_ids = sorted(TASK_SETS[task_set])
    rng.shuffle(instruction_ids)
    results: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    attempts = 0
    while len(results) < count:
        if attempts > max(10_000, count * 100):
            raise RuntimeError("Không thể tạo đủ mẫu không trùng lặp.")
        instruction_id = instruction_ids[attempts % len(instruction_ids)]
        base_prompt = prompts[(attempts // len(instruction_ids)) % len(prompts)]
        example = _make_example(
            instruction_id, base_prompt, key_offset + len(results), rng,
            task_set)
        attempts += 1
        if example["prompt"] in seen_prompts:
            continue
        seen_prompts.add(example["prompt"])
        results.append(example)
    rng.shuffle(results)
    return results


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def generate_dataset(*, output_dir: Path, prompts: list[str], train_size: int,
                     val_size: int, seed: int,
                     task_set: str = "core") -> dict[str, Any]:
    if train_size < 1 or val_size < 1:
        raise ValueError("train-size và val-size phải lớn hơn 0.")
    if task_set not in TASK_SETS:
        raise ValueError(
            f"task_set phải là một trong {sorted(TASK_SETS)}, nhận được {task_set!r}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sinh một tập duy nhất không trùng rồi mới chia để không rò rỉ câu hỏi qua
    # train/validation, kể cả với các nhiệm vụ có mô tả cố định.
    examples = _generate_examples(
        train_size + val_size, prompts, seed, 0, task_set)
    train = examples[:train_size]
    val = examples[train_size:]

    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)
    counts = Counter(
        instruction_id
        for row in train + val
        for instruction_id in row["instruction_id_list"])
    manifest = {
        "language": "vi",
        "task_set": task_set,
        "seed": seed,
        "train_size": len(train),
        "val_size": len(val),
        "base_prompt_count": len(prompts),
        "available_instruction_count": len(TASK_SETS[task_set]),
        "instruction_ids": sorted(counts),
        "instruction_counts": dict(sorted(counts.items())),
        "experimental_ids_included": False,
        "format": "IFBench fields plus messages and verifier metadata",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo tập train/validation tiếng Việt cho RL với IFBench.")
    parser.add_argument(
        "--task-set", choices=sorted(TASK_SETS), default="core",
        help=("core: 39 tác vụ đã duyệt; hard: 13 tác vụ khó; "
              "all: gộp cả hai nhóm."),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Mặc định: data/vi_rl_<task-set>.",
    )
    parser.add_argument("--base-prompts", type=Path,
                        help="Tệp TXT (mỗi dòng một câu) hoặc JSONL; không bắt buộc.")
    parser.add_argument("--prompt-field", default="prompt",
                        help="Tên trường câu hỏi khi --base-prompts là JSONL.")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f"data/vi_rl_{args.task_set}")
    prompts = _load_prompts(args.base_prompts, args.prompt_field)
    manifest = generate_dataset(
        output_dir=args.output_dir,
        prompts=prompts,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        task_set=args.task_set,
    )
    print(
        f"Đã tạo {manifest['train_size']} mẫu train và "
        f"{manifest['val_size']} mẫu validation trong {args.output_dir}.")
    print(
        f"Nhóm tác vụ: {manifest['task_set']} "
        f"({manifest['available_instruction_count']} tác vụ khả dụng; "
        f"{len(manifest['instruction_ids'])} tác vụ xuất hiện).")


if __name__ == "__main__":
    main()
