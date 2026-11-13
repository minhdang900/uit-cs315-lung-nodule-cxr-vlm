from nodulecxr.prompts import ANSWER_PREFIX, QUESTION_LOC, build_target, parse_answer


def test_question_matches_spec():
    assert QUESTION_LOC == ("Is there a lung nodule or mass on this chest X-ray? If yes, give the "
                            "bounding box(es) in pixel coordinates. Answer in JSON.")


def test_target_format_and_prefix():
    t = build_target(1, [[10.4, 20.6, 30.0, 40.0]])
    assert t == '{"nodule": "yes", "boxes": [[10, 21, 30, 40]]}' and t.startswith(ANSWER_PREFIX)
    assert build_target(0, []) == '{"nodule": "no", "boxes": []}'
    assert build_target(1, [[1, 2, 3, 4]], with_boxes=False) == '{"nodule": "yes"}'


def test_roundtrip_and_fences():
    p = parse_answer("```json\n" + build_target(1, [[10, 20, 30, 40]]) + "\n```")
    assert p.ok and p.label == 1 and p.boxes == [[10.0, 20.0, 30.0, 40.0]]


def test_clip_drop_and_qwen_format():
    p = parse_answer('{"nodule": "yes", "boxes": [[-5, 10, 2000, 50], [5, 5, 5.5, 9]]}')
    assert p.boxes == [[0.0, 10.0, 896.0, 50.0]]
    q = parse_answer('{"nodule": "yes", "boxes": [{"bbox_2d": [1, 2, 30, 40], "label": "nodule"}]}')
    assert q.boxes == [[1.0, 2.0, 30.0, 40.0]]


def test_no_drops_boxes_and_fallbacks():
    assert parse_answer('{"nodule": "no", "boxes": [[1,2,3,4]]}').boxes == []
    f = parse_answer('Answer: nodule: yes, right upper lobe')
    assert f.label == 1 and not f.ok
    g = parse_answer("No acute cardiopulmonary abnormality.")
    assert g.label is None and not g.ok
