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


def test_boolean_and_string_variants():
    assert parse_answer('{"nodule": true, "boxes": [[1, 2, 30, 40]]}').label == 1
    assert parse_answer('{"nodule": false}').label == 0
    for v, lab in (('"Yes."', 1), ('"NO"', 0), ('"true"', 1), ('"False!"', 0)):
        p = parse_answer('{"nodule": %s}' % v)
        assert p.ok and p.label == lab, v


def test_nodule_prefixed_keys():
    p = parse_answer('{"nodule_present": "yes", "boxes": [[1, 2, 30, 40]]}')
    assert p.ok and p.label == 1 and p.boxes == [[1.0, 2.0, 30.0, 40.0]]
    assert parse_answer('{"nodule_or_mass": false}').label == 0


def test_alternative_box_keys():
    for k in ("boxes", "bounding_boxes", "bbox", "bboxes"):
        p = parse_answer('{"nodule": "yes", "%s": [[1, 2, 30, 40]]}' % k)
        assert p.boxes == [[1.0, 2.0, 30.0, 40.0]], k
    assert parse_answer('{"nodule": "yes", "bbox": [1, 2, 30, 40]}').boxes == [[1.0, 2.0, 30.0, 40.0]]


def test_prose_json_and_trailing_braces():
    p = parse_answer('Sure. {"nodule": "yes", "boxes": [[10, 20, 30, 40]]} Hope this helps {}')
    assert p.ok and p.label == 1 and p.boxes == [[10.0, 20.0, 30.0, 40.0]]
    q = parse_answer('{"note": "x"} then {"nodule": "no"}')
    assert q.ok and q.label == 0


def test_regex_fallback_variants():
    f = parse_answer('nodule_present: true, but the JSON is cut {"nodule_present": tr')
    assert f.label == 1 and not f.ok
    assert parse_answer('"nodule": false (truncated').label == 0


def test_non_label_nodule_key_does_not_shadow_label_key():
    p = parse_answer('{"nodule_location": "RUL", "nodule_present": "yes", "boxes": [[10, 10, 50, 50]]}')
    assert p.ok and p.label == 1 and p.boxes == [[10.0, 10.0, 50.0, 50.0]]
    q = parse_answer('{"nodule_size_mm": 12, "nodule_present": true}')
    assert q.ok and q.label == 1
    assert parse_answer('{"nodule_location": "right upper lobe"}').label is None
