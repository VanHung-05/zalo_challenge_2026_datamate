import pytest
import pandas as pd
from src.datamate.entity_linker import extract_candidate_phrases, link_entities
from src.datamate.profiler import build_value_index


def test_extract_candidate_phrases():
    q = 'xem giúp ngày 25/09/2025 cửa hàng bán Kẹo Alpenliebe 100g giá bao nhiêu một gói'
    cands = extract_candidate_phrases(q)
    assert any("Kẹo Alpenliebe" in c or "Alpenliebe" in c for c in cands)


def test_link_entities_match():
    tables = {
        "BanHang": pd.DataFrame({
            "Tên sản phẩm": ["Kẹo Alpenliebe 100g", "Bột ngọt Ajinomoto 400g", "Trà xanh C2"],
            "Nhóm hàng": ["Bánh kẹo", "Gia vị", "Nước giải khát"]
        })
    }
    v_index = build_value_index(tables)
    matches = link_entities("Cửa hàng còn Kẹo Alpenliebe không?", v_index)
    assert len(matches) > 0
    assert matches[0].matched_value == "Kẹo Alpenliebe 100g"
    assert matches[0].column == "Tên sản phẩm"
    assert matches[0].score >= 0.8
