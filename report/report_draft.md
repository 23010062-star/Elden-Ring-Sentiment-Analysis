# BÁO CÁO ĐỒ ÁN KHAI PHÁ DỮ LIỆU

## Phân tích cảm xúc đánh giá người chơi Elden Ring trên Steam

### Tóm tắt

Đồ án xây dựng bài toán phân loại nhị phân trên các đánh giá Steam của trò chơi *Elden Ring*. Văn bản đánh giá (`review`) là đầu vào; trường khuyến nghị Steam (`voted_up`) được dùng làm nhãn gần đúng: 1 là đề xuất trò chơi, 0 là không đề xuất. Ba cách biểu diễn văn bản được so sánh với Logistic Regression: TF-IDF, embedding từ `sentence-transformers/all-MiniLM-L6-v2`, và embedding từ `Snowflake/snowflake-arctic-embed-s`. Sau khi xử lý trùng lặp và nhãn mâu thuẫn, tập dữ liệu gồm 7.946 review, chia thành train 5.562, validation 1.192 và test 1.192. Cấu hình TF-IDF + Logistic Regression có kết quả Test cao nhất theo Macro-F1 (0,7124), kế đến Arctic Embed S (0,6929) và MiniLM (0,6714). Kết quả phản ánh khả năng dự đoán nhãn khuyến nghị, không phải phép đo hoàn hảo về cảm xúc chủ quan trong câu chữ.

---

## Chương 1. Tổng quan đề tài và bài toán

### 1.1. Bối cảnh

Đánh giá trên nền tảng trò chơi trực tuyến cung cấp cả nội dung nhận xét và lựa chọn khuyến nghị của người chơi. Khai phá nội dung này giúp khảo sát cách biểu diễn văn bản ảnh hưởng đến việc dự đoán phản hồi của người dùng. Đồ án tập trung vào đánh giá tiếng Anh về *Elden Ring* được công bố trên Steam trong một khoảng thời gian giới hạn năm 2023 [1].

### 1.2. Mục tiêu

Mục tiêu là xây dựng và so sánh ba pipeline phân loại văn bản, sử dụng cùng tập Train, Validation và Test:

1. TF-IDF kết hợp Logistic Regression.
2. Sentence embedding từ MiniLM kết hợp Logistic Regression.
3. Sentence embedding từ Snowflake Arctic Embed S kết hợp Logistic Regression.

Mỗi pipeline được thử với Logistic Regression mặc định (các tham số mô hình giữ nguyên, `max_iter=1000`) và `class_weight="balanced"`. Cấu hình được lựa chọn trên Validation theo Macro-F1; đánh giá Test chỉ được thực hiện sau khi lựa chọn và không dùng để điều chỉnh mô hình.

### 1.3. Phát biểu bài toán

Cho văn bản đánh giá \(x\), dự đoán nhãn nhị phân \(y\):

- \(y=1\): người viết chọn “đề xuất” trò chơi (`voted_up=True`).
- \(y=0\): người viết không đề xuất (`voted_up=False`).

Nhãn này là tín hiệu khuyến nghị của Steam, được dùng làm đại diện cho cảm xúc tích cực/không tích cực. Nó không tương đương hoàn toàn với nhãn cảm xúc được gán thủ công cho nội dung: người chơi có thể đề xuất trò chơi nhưng vẫn nêu nhiều điểm chưa hài lòng, hoặc ngược lại.

---

## Chương 2. Cơ sở lý thuyết và các phương pháp

### 2.1. TF-IDF

TF-IDF biểu diễn văn bản bằng trọng số của từng từ. Thành phần TF phản ánh tần suất từ trong văn bản; IDF giảm trọng số của từ xuất hiện ở nhiều văn bản. Dạng khái quát:

\[
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\times
\left(\log\frac{1+N}{1+\operatorname{df}(t)}+1\right),
\]

trong đó \(t\) là từ, \(d\) là văn bản, \(N\) là số văn bản và \(\operatorname{df}(t)\) là số văn bản chứa từ đó. Đồ án dùng `TfidfVectorizer()` mặc định của scikit-learn: unigram, chuẩn hóa L2 và IDF có làm trơn; từ vựng và IDF chỉ được học trên Train [2, 3]. Cách biểu diễn này đơn giản, nhanh và dễ diễn giải qua các từ đặc trưng, nhưng không thể hiện trực tiếp ngữ cảnh hay quan hệ ngữ nghĩa sâu giữa các từ.

### 2.2. Logistic Regression

Logistic Regression ước lượng xác suất thuộc lớp 1 từ tổ hợp tuyến tính các đặc trưng:

\[
P(y=1\mid x)=\sigma(w^Tx+b),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

Mô hình dự đoán lớp dựa trên xác suất/ranh giới quyết định và được huấn luyện bằng cách tối ưu hàm mất mát logistic có điều chuẩn [4]. Đây là bộ phân loại tuyến tính có thể làm việc với vector TF-IDF thưa cũng như embedding đặc. Tất cả lần huấn luyện trong đồ án dùng `max_iter=1000`. Với `class_weight="balanced"`, trọng số mỗi lớp được tính nghịch đảo theo tần suất lớp trong Train; phương án này tăng ảnh hưởng của lớp ít mẫu mà không nhân bản hay loại mẫu [4].

### 2.3. Sentence embedding và MiniLM

Sentence embedding ánh xạ câu thành vector đặc, trong đó các biểu diễn được huấn luyện trước có thể mã hóa thông tin ngữ nghĩa của toàn câu. Sentence-BERT sử dụng kiến trúc hai nhánh/siamese để tạo vector câu có thể so sánh hiệu quả [5].

Trong đồ án sử dụng `sentence-transformers/all-MiniLM-L6-v2`, mô hình tiếng Anh tạo embedding 384 chiều. Mô hình có sẵn được nạp và dùng để mã hóa review; không fine-tune trên đồ án. Sau đó Logistic Regression được huấn luyện trên embedding Train. MiniLM bắt được ngữ cảnh tốt hơn biểu diễn đếm từ nhưng cần suy luận Transformer cho mỗi văn bản; model card cũng nêu giới hạn đầu vào 256 word pieces, vì vậy văn bản dài có thể bị cắt [6, 7].

### 2.4. Snowflake Arctic Embed S

`Snowflake/snowflake-arctic-embed-s` là mô hình embedding nhỏ trong họ Arctic Embed, được thiết kế chủ yếu cho truy hồi văn bản. Tài liệu mô hình công bố embedding 384 chiều và khuyến nghị `prompt_name="query"` cho văn bản truy vấn [8, 9]. Đồ án dùng prompt này cho review ở cả Train và Validation/Test, giữ nguyên embedding đã huấn luyện trước và chỉ fit Logistic Regression trên Train. Ưu điểm là biểu diễn ngữ nghĩa dày; hạn chế là mục tiêu tiền huấn luyện truy hồi không hoàn toàn trùng với phân loại cảm xúc.

### 2.5. Chỉ số đánh giá

Precision, Recall và F1-score trong bảng được tính cho lớp 1. F1 là trung bình điều hòa giữa Precision và Recall. Macro-F1 là trung bình số học F1 của hai lớp, do đó cho mỗi lớp trọng số như nhau. F1 lớp 0 được báo cáo riêng để phản ánh khả năng nhận diện đánh giá không đề xuất. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán [10]. Do dữ liệu mất cân bằng, Accuracy được xem cùng Macro-F1 và F1 lớp 0, không dùng riêng lẻ để kết luận.

---

## Chương 3. Bộ dữ liệu, EDA và tiền xử lý

### 3.1. Nguồn và cấu trúc

Dữ liệu được lấy từ bộ [Elden Ring Steam Reviews trên Kaggle](https://www.kaggle.com/datasets/noahx1/elden-ring-steam-reviews) [1]. Tệp CSV gốc có 9.794 dòng và 16 cột, dung lượng 2.624.148 byte. Theo mô tả của tác giả, bộ dữ liệu gồm review tiếng Anh được thu thập từ ngày 29/01/2023 đến ngày 07/03/2023. Các trường gồm nội dung review, nhãn khuyến nghị, thời điểm, phiếu hữu ích, thông tin mua/nhận miễn phí, thời gian chơi và một số thống kê tác giả.

### 3.2. Khám phá dữ liệu

EDA cho thấy:

- Có 31 review thiếu nội dung: 29 ô NA và 2 chuỗi chỉ chứa khoảng trắng. Các cột khác không thiếu.
- Không có dòng trùng hoàn toàn hoặc mã `id` trùng trong CSV gốc; không có giá trị âm ở các bộ đếm/thời lượng đã kiểm tra và không có thời điểm sai định dạng.
- Có 12 bản ghi mà thời gian chơi trong hai tuần gần nhất lớn hơn tổng thời gian chơi. Đây là bất nhất trong metadata, được ghi nhận nhưng không dùng làm đầu vào mô hình.
- Trong 9.794 nhãn ban đầu, có 9.201 `True` (93,95%) và 593 `False` (6,05%), tương đương khoảng 15,52:1.
- Sau khi bỏ review trống, 9.763 review có nội dung được dùng để khảo sát. Độ dài trung vị là 31 ký tự (6 từ), trong khi trung bình là 155,72 ký tự (28,4 từ), cho thấy phân phối lệch phải với một số review rất dài.

Notebook EDA trực quan hóa phân bố nhãn, độ dài review, độ dài theo nhãn và số review theo ngày. Do dữ liệu chỉ bao phủ 38 ngày, xu hướng ngày không thể đại diện cho toàn bộ lịch sử đánh giá của trò chơi.

### 3.3. Làm sạch và nhãn

31 review trống được loại khỏi tập phân tích văn bản. Làm sạch ở mức cơ bản gồm giải mã HTML entity, bỏ thẻ HTML nếu có và chuẩn hóa khoảng trắng. Không xóa dấu câu, không chuyển chữ thường ở bước làm sạch và không loại các từ phủ định như `not`, `no`, `never`. `voted_up` được ánh xạ thành `label`: True → 1, False → 0. Chỉ `review` và `label` được giữ cho phân loại; các metadata khác không đưa vào làm đặc trưng.

### 3.4. Xử lý trùng lặp và chia tập

Sau lần chia ban đầu, kiểm tra nội dung phát hiện review trùng giữa các tập (152 review giữa Train–Validation, 134 giữa Train–Test và 74 giữa Validation–Test, theo so khớp chính xác). Vì cùng một văn bản có thể khiến đánh giá quá lạc quan, dữ liệu được gộp lại và gom nhóm theo nội dung không phân biệt hoa/thường — phù hợp với cách TF-IDF mặc định chuyển chữ thường.

Trong 9.763 hàng, có 7.964 nội dung duy nhất theo khóa đã chuẩn hóa. 18 nhóm gồm 234 hàng có nhãn mâu thuẫn bị loại vì không thể xác định nhãn thống nhất. Trong 9.529 hàng còn lại, 1.583 hàng là bản lặp; giữ một hàng cho mỗi nội dung nhất quán, còn 7.946 mẫu. Như vậy, phần giảm từ 9.763 xuống 7.946 được giải thích đầy đủ bởi việc loại các nhóm nhãn mâu thuẫn và các bản lặp, không phải mất dữ liệu ngoài dự kiến.

Các nội dung review sau đó được chia lại phân tầng với `random_state=42`, theo tỷ lệ xấp xỉ 70%/15%/15%. Không cân bằng mẫu trước khi chia.

| Tập | Số mẫu | Không đề xuất (0) | Đề xuất (1) | Tỷ lệ lớp 1 |
|---|---:|---:|---:|---:|
| Train | 5.562 | 394 | 5.168 | 92,92% |
| Validation | 1.192 | 84 | 1.108 | 92,95% |
| Test | 1.192 | 85 | 1.107 | 92,87% |
| **Tổng** | **7.946** | **563** | **7.383** | **92,91%** |

Các review được kiểm tra không giao nhau giữa các tập sau bước xử lý. `class_weight="balanced"` chỉ được áp dụng trong quá trình fit các mô hình cần cân bằng ảnh hưởng lớp; không thay đổi các CSV.

---

## Chương 4. Thực nghiệm, kết quả và so sánh

### 4.1. Thiết lập thực nghiệm

Thực nghiệm chạy trong môi trường ảo `.venv`, trên CPU. TF-IDF được fit trên Train và chỉ transform các tập đánh giá. Hai mô hình Transformer pretrained tạo embedding 384 chiều; không fine-tune. Logistic Regression được huấn luyện với `max_iter=1000`. Trên Validation, mỗi phương pháp so sánh cấu hình mặc định với `class_weight="balanced"`; chọn cấu hình có Macro-F1 cao hơn. Cả ba phương pháp đều chọn cấu hình balanced để đánh giá cuối cùng.

Trong lần chạy ghi ở notebook, MiniLM mã hóa Train và Validation lần lượt trong 20,25 và 4,09 giây; Arctic Embed S mất 54,25 và 12,85 giây. Thời gian nạp mô hình pretrained lần lượt là 17,35 và 24,57 giây. Thời gian `fit` Logistic Regression balanced được ghi nhận là 0,066 giây cho TF-IDF, 0,0146 giây cho MiniLM và 0,0171 giây cho Arctic Embed S. Đây là thời gian của một lần chạy CPU cụ thể, không phải benchmark phần cứng; thời gian nạp mô hình và tạo embedding được tách khỏi thời gian fit bộ phân loại.

### 4.2. Kết quả Validation và lựa chọn cấu hình

| Phương pháp | Cấu hình | Accuracy | Macro-F1 | F1 lớp 0 |
|---|---|---:|---:|---:|
| TF-IDF | Mặc định | 0,9304 | 0,5049 | 0,0460 |
| TF-IDF | Balanced | 0,8842 | 0,6827 | 0,4298 |
| MiniLM | Mặc định | 0,9421 | 0,6433 | 0,3168 |
| MiniLM | Balanced | 0,8423 | 0,6589 | 0,4088 |
| Arctic Embed S | Mặc định | 0,9413 | 0,6415 | 0,3137 |
| Arctic Embed S | Balanced | 0,8691 | **0,6920** | **0,4583** |

Trong từng họ mô hình, balanced có Macro-F1 cao hơn mặc định. Cấu hình Arctic Embed S balanced đạt Macro-F1 Validation cao nhất. Dù vậy, sự khác biệt giữa các kết quả còn phụ thuộc vào tập đánh giá; quyết định cuối được kiểm tra một lần trên Test theo kế hoạch.

### 4.3. Đánh giá cuối cùng trên Test

Các pipeline và tham số được giữ nguyên sau Validation. TF-IDF vocabulary/IDF và cả ba Logistic Regression chỉ được fit trên Train. Embedding của Test chỉ được tạo để dự đoán; Test không được dùng để chọn mô hình. Precision, Recall và F1-score trong bảng là của lớp 1.

| Phương pháp (balanced) | Accuracy | Precision | Recall | F1-score | Macro-F1 | F1 lớp 0 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0,8943 | 0,9748 | 0,9097 | 0,9411 | **0,7124** | **0,4836** |
| MiniLM + Logistic Regression | 0,8456 | 0,9843 | 0,8473 | 0,9107 | 0,6714 | 0,4321 |
| Snowflake Arctic Embed S + Logistic Regression | 0,8641 | **0,9846** | 0,8672 | 0,9222 | 0,6929 | 0,4636 |

TF-IDF balanced đạt Macro-F1 và F1 lớp 0 cao nhất trên Test; Arctic Embed S đứng thứ hai theo cả hai chỉ số, còn MiniLM đứng thứ ba. Arctic có Precision lớp 1 cao nhất (0,9846), trong khi TF-IDF có Recall lớp 1 cao nhất (0,9097). Confusion matrix của ba phương pháp được trình bày trong notebook `06_final_evaluation.ipynb`.

### 4.4. Validation so với Test

| Phương pháp | Macro-F1 Validation | Macro-F1 Test | F1 lớp 0 Validation | F1 lớp 0 Test |
|---|---:|---:|---:|---:|
| TF-IDF balanced | 0,6827 | 0,7124 | 0,4298 | 0,4836 |
| MiniLM balanced | 0,6589 | 0,6714 | 0,4088 | 0,4321 |
| Arctic Embed S balanced | 0,6920 | 0,6929 | 0,4583 | 0,4636 |

Macro-F1 và F1 lớp 0 của cả ba phương pháp cao hơn đôi chút trên Test so với Validation, nhưng mức chênh lệch không lớn (Macro-F1 tăng 0,0009–0,0297). Accuracy Test của TF-IDF và MiniLM cũng cao hơn Validation (lần lượt 0,8943 so với 0,8842 và 0,8456 so với 0,8423), trong khi Arctic giảm nhẹ (0,8641 so với 0,8691). Validation chỉ có 84 mẫu lớp 0 và Test có 85; dao động ở nhóm nhỏ này có thể ảnh hưởng đáng kể đến F1 lớp 0. Không thực hiện kiểm định thống kê, vì vậy không kết luận khác biệt là có ý nghĩa thống kê.

Kết quả cho thấy Accuracy cao không đồng nghĩa phát hiện tốt lớp ít mẫu. Với lớp 0 chỉ chiếm xấp xỉ 7,1%, Macro-F1 và F1 lớp 0 giúp nhìn rõ hơn sự đánh đổi. `class_weight="balanced"` tăng khả năng nhận diện lớp 0 so với cấu hình mặc định trong các lần thử Validation, dù Accuracy thường giảm.

---

## Chương 5. Kết luận, hạn chế và hướng phát triển

### 5.1. Kết luận

Đồ án đã xây dựng ba pipeline dùng cùng cách chia dữ liệu không giao nhau theo nội dung review và đánh giá trên cùng tập Test. Cấu hình có Macro-F1 Test cao nhất là **TF-IDF + Logistic Regression balanced (0,7124)**, đồng thời đạt F1 lớp 0 cao nhất (0,4836). Kết quả này cho thấy trên bộ dữ liệu và thiết lập hiện tại, biểu diễn từ vựng đơn giản vẫn cạnh tranh với hai embedding Transformer. Arctic Embed S balanced đạt Macro-F1 0,6929 và MiniLM balanced đạt 0,6714. Không thể suy rộng thứ hạng này cho các bộ dữ liệu hoặc cách gán nhãn khác.

### 5.2. Hạn chế

1. `voted_up` là lựa chọn khuyến nghị của người dùng, không phải nhãn cảm xúc được gán thủ công cho văn bản; do đó đây là bài toán dự đoán khuyến nghị với nhãn đại diện.
2. Dữ liệu chỉ gồm review tiếng Anh trong khoảng 29/01–07/03/2023 và có thể không đại diện cho người chơi ở thời điểm hoặc ngôn ngữ khác.
3. Nhãn mất cân bằng mạnh; lớp 0 còn 563 mẫu sau xử lý. Dù dùng class weight, F1 lớp 0 trên Test vẫn dưới 0,5.
4. Việc loại 234 hàng thuộc nhóm review có nhãn mâu thuẫn và gộp các nội dung trùng tránh rò rỉ giữa tập, nhưng cũng làm mất một phần số lần lặp/khác biệt nhãn có thể phản ánh ý kiến thực tế.
5. Kết quả dựa trên một lần chia cố định; không có cross-validation, khoảng tin cậy hoặc kiểm định ý nghĩa thống kê. Embedding phụ thuộc mô hình pretrained và giới hạn độ dài đầu vào.

### 5.3. Hướng phát triển

Các hướng có thể xem xét trong nghiên cứu tiếp theo gồm: thu thập thêm dữ liệu và nhãn cảm xúc được gán/kiểm tra thủ công; phân tích lỗi riêng trên lớp 0; đánh giá độ ổn định qua nhiều lần chia hoặc cross-validation có nhóm theo nội dung/người đánh giá; và khảo sát ảnh hưởng của review dài bị cắt khi mã hóa Transformer. Các bước này nằm ngoài phạm vi thực nghiệm hiện tại.

---

## Tài liệu tham khảo

[1] Noahx1. *Elden Ring Steam Reviews*. Kaggle, 2023. https://www.kaggle.com/datasets/noahx1/elden-ring-steam-reviews (mô tả bộ dữ liệu và nguồn dữ liệu).

[2] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0.

[3] scikit-learn developers. *TfidfVectorizer — API Reference*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.

[4] scikit-learn developers. *LogisticRegression — API Reference*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

[5] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*. https://doi.org/10.48550/arXiv.1908.10084.

[6] Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. https://doi.org/10.48550/arXiv.2002.10957.

[7] Sentence Transformers. *sentence-transformers/all-MiniLM-L6-v2 — Model Card*. Hugging Face. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.

[8] Merrick, L., Xu, D., Nuti, G., & Campos, D. (2024). Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models. https://doi.org/10.48550/arXiv.2405.05374.

[9] Snowflake. *Snowflake/snowflake-arctic-embed-s — Model Card*. Hugging Face. https://huggingface.co/Snowflake/snowflake-arctic-embed-s.

[10] scikit-learn developers. *Metrics and scoring: quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics.
