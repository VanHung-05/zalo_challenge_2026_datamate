from src.datamate.text_norm import btc_normalize, normalize_question, strip_accents


def test_btc_normalize():
    assert btc_normalize("Đàm Hà Duy - 0932 811 434") == "đàm hà duy 0932 811 434"
    assert btc_normalize("  Sting dâu 330ml!! ") == "sting dâu 330ml"
    assert btc_normalize("SP0042") == "sp0042"


def test_strip_accents():
    assert strip_accents("Đường Hai Bà Trưng") == "Duong Hai Ba Trung"
    assert strip_accents("≥ 8 đơn") == "≥ 8 don"


def test_telex_repair():
    text, _ = normalize_question("coi giumf tieenf bán Trà vải (L) tháng 3/2025")
    assert "giùm tiền" in text
    text, _ = normalize_question("số nợ cuar từng nguwowif, ghi lần luwowtj vaayj")
    assert "của từng người" in text and "lần lượt vậy" in text
    text, _ = normalize_question("heets hoom nay doanh thu thangs 5 tawng hay giamr")
    assert text == "hết hôm nay doanh thu tháng 5 tăng hay giảm"


def test_teencode():
    text, ascii_text = normalize_question("dthu tuần r bnhiu, ko dc thì th trước")
    assert text == "doanh thu tuần rồi bao nhiêu, không được thì tháng trước"
    assert ascii_text == "doanh thu tuan roi bao nhieu, khong duoc thi thang truoc"
    assert normalize_question("mua nhiu tiền nhất")[0] == "mua nhiều tiền nhất"
    assert normalize_question("có nhiu ngày")[0] == "có nhiêu ngày"


def test_product_words_untouched():
    # 'TH' brand, '250k' amount and codes are not teencode
    text, _ = normalize_question("Sữa TH true MILK 180ml giá 250k mã SP0042")
    assert text == "sữa th true milk 180ml giá 250k mã sp0042"
