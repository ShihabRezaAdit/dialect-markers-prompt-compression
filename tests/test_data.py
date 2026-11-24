import numpy as np

from idea86 import data as D


def fake_records(n_par=500):
    recs = []
    for p in range(n_par):
        nq = 1 + (p % 4)
        for j in range(nq):
            recs.append({"id": f"q{p}_{j}", "title": f"T{p // 10}", "context": f"context {p}",
                         "question": f"question {p} {j}?", "answers": {"text": [f"ans{p}{j}"], "answer_start": [0]}})
    return recs


def test_paragraph_grouping_and_order():
    paras = D.squad_paragraphs(fake_records(20))
    assert [p["pnum"] for p in paras] == list(range(20))
    assert len(paras[3]["questions"]) == 4 and paras[3]["questions"][0]["qid"] == "q3_0"


def test_pilot_sample_fixed_and_disjoint():
    paras = D.squad_paragraphs(fake_records())
    a, b = D.pilot_sample(paras), D.pilot_sample(paras)
    assert a == b
    assert len(a["passages"]) == 300 and len(a["questions"]) == 600
    assert len(a["smoke_passages"]) == 10 and len(a["smoke_questions"]) == 20
    pil = {p["pnum"] for p in a["passages"]}
    assert not pil & {p["pnum"] for p in a["smoke_passages"]}
    assert all(len(paras[p["pnum"]]["questions"]) >= 2 for p in a["passages"])
    assert a["questions"][5]["ppos"] == 2 and a["questions"][5]["pos"] == 5


def test_audit_sample():
    s = D.audit_sample()["pairs"]
    assert len(s) == 100 and len({p["ppos"] for p in s}) == 100
    assert sum(p["dialect"] == "A" for p in s) == 50
    assert all(p["qpos"] // 2 == p["ppos"] for p in s)


def test_real_aae_sample():
    pilot = D.real_aae_sample(17479, 1, 200)
    study = D.real_aae_sample(17479, 201, 2200)
    assert len(pilot) == 200 and len(study) == 2000 and not set(pilot) & set(study)
    assert pilot == [int(i) for i in np.random.default_rng(86).permutation(17479)[:200]]


def test_conllu_and_fallback():
    txt = "# sent_id = 1\n# text = yall be trippin fr\n1\tyall\t_\n\n# sent_id = 2\n# text = ok then\n1\tok\n"
    assert D.read_conllu_texts(txt) == ["yall be trippin fr", "ok then"]
    p, s = D.ud_fallback_sample([str(i) for i in range(250)])
    assert len(p) == 50 and len(s) == 200 and not set(p) & set(s)
