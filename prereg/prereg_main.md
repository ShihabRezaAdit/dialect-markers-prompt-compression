# Main-study pre-commitment 1 of 2: must-have experiments (Idea 86)

Written 2026-11-28T01:08:09Z at git commit `cc84c58287363929de06a99d114a1572bd579189`, before any main-study text was rendered, compressed or scored. The configuration, prompts, item lists, hypotheses, families and the pipeline below are frozen; any change is logged with its reason in the decisions log and reported.

This is the first of two main-study pre-commitments (decisions log: Gate 1 change). It records the SHA-256 of the internal execution plan's source (plan_sources_sha256 below), so the E3 hypotheses, the H3 rule, the four answer models and their pinned revisions, the prompt templates and every threshold are frozen by the plan text from this commit on. The second, generation pre-commitment freezes the E3 code and the result of the Mistral-7B-Instruct-v0.3 and Gemma-2-9B-it setup check; it is committed before any E3 answer is generated, and no accuracy or flip analysis of main-study data takes place before it.

Confirmatory dialect: AAVE. Families: F1 (three Delta_s tests, Holm) on SQuAD at keep rate 0.5, with REAL-AAE as the replication family; F2 to F4 are confirmatory only if their experiment is run. H3 is supported if at least 6 of the 12 AAVE accuracy cells widen the gap significantly and none narrows it significantly; accuracy is measured on answer-preserved questions, with all questions as a secondary analysis.

```json
{
 "meta": {
  "git_commit": "cc84c58287363929de06a99d114a1572bd579189",
  "unit_tests": {
   "code_hash": "e964f4e8146d7b94043326352f181121b78b6e70009713c311f1d35fc5e2fcc6",
   "git_commit": "cc84c58287363929de06a99d114a1572bd579189",
   "passed": true,
   "utc": "2026-11-28T00:09:35Z"
  },
  "written_utc": "2026-11-28T01:08:09Z"
 },
 "state": {
  "code_hash": "e964f4e8146d7b94043326352f181121b78b6e70009713c311f1d35fc5e2fcc6",
  "config_hash": "8a3455b90b5bf19ab3c850268cf21d8a87a3a8ce33ea48b3e21ac1d608ed6eaa",
  "data": {
   "meetingbank": {
    "file": "data/train-00000-of-00001.parquet",
    "repo": "microsoft/MeetingBank-LLMCompressed",
    "revision": "9b8098be677863e6f8248085d2ba714d2b7845b7",
    "rows": 5169,
    "sha256": "ca975d4fa9e36ee192f4305ea0a0fc58f1879ae7b056c932816a857508644d58"
   },
   "real_aae": {
    "rows": 17479,
    "sha256": "7dba90985790315ede8fe9d5494408ce8355d74437d5e98cea2789fafc09e5bc",
    "study_rows": 17279
   },
   "squad": {
    "config": "plain_text",
    "paragraphs": 2067,
    "questions": 10570,
    "repo": "rajpurkar/squad",
    "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "split": "validation"
   }
  },
  "env_locks_sha256": {
   "env-compress": "b1f62ac32a9560096bf82f1c386226b74fb988d32106b3bdc7e185f971dc7eb1",
   "env-gen": "dfadda9dc108fc267c8f2162e5cff2d8dc38ed2dd8fb2bc5e18474c9ae429630",
   "env-render": "a68d547098c4dfcf8607bbaa2cb8e652cba6bf8ab2e494759039317b1d21ccb9",
   "env-sc": "15a188f9feaec676bc74447c7b92363a72e33372d28551966216bf94beecd052",
   "env-stats": "0f02f72148a7d518d56c835570f55f1aa2a7601141f16ca224206aced9a16da0"
  },
  "hypotheses": {
   "F1": {
    "cells": [
     "L1",
     "L2",
     "SC"
    ],
    "corpus": "squad",
    "correction": "holm",
    "measure": "delta_s",
    "rate": 0.5
   },
   "F2_F4": "confirmatory only if their should-have experiment is run (E3, E4)",
   "H1a": {
    "compressor": "L1",
    "direction": "positive"
   },
   "H1b": {
    "compressor": "L2",
    "direction": "two-sided",
    "expected": "negative"
   },
   "H1c": {
    "compressor": "R",
    "rule": "90% interval inside +-0.03"
   },
   "H1d": {
    "compressor": "SC",
    "direction": "two-sided"
   },
   "H3": {
    "cells": "12 AAVE accuracy cells (delta_gamma)",
    "primary_set": "questions answer-preserved in AAVE",
    "rule": "supported if at least 6 cells significantly widen the gap and none significantly narrows it",
    "secondary_set": "all questions"
   },
   "replication_real_aae": {
    "cells": [
     "L1",
     "L2",
     "SC"
    ],
    "correction": "holm",
    "measure": "delta_s",
    "rate": 0.5
   }
  },
  "item_lists_sha256": {
   "audit": "69ca889b54f0428bfa69f1d8bd16eeae147dc37d3ac16f0ae5bb8365059f0e45",
   "passages": "38b3be8eb99dea97c6dbf54896c799dca2e0ec101665bc4229f2e4dd95067a7a",
   "questions": "79040e25411da788a27a96b78958e4a51cc86b647b485827b05b06adb000827e",
   "tweet_rows": "caafe451b6493c6010264cb1c81346a4706272afa8623d63a663f99f472dd8d6"
  },
  "model_revisions": {
   "gpt2": {
    "repo": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"
   },
   "llama2": {
    "repo": "NousResearch/Llama-2-7b-hf",
    "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
   },
   "xlmr": {
    "repo": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "revision": "ebaba9b0e874dadd3003ffcff828e4397e568089"
   }
  },
  "plan_sources_sha256": {
   "annotation_guide_pdf": "db6115222a1976e8b3fb78726e1a2c1f016bb79dded43c5ab8d2f97626d486a0",
   "internal_plan_tex": "aeb66381b7c93009663f629b7481c6caaf51d4309414739c914710c0799dbf31",
   "professor_proposal_tex": "52bfe02e288321c9dbe26f26260f31edf9cca459a87aeee21236663405f23e84"
  },
  "priorities": {
   "must": [
    "E1 AAVE rendering and 150-pair audit",
    "E2 SQuAD AAVE and REAL-AAE retention at three rates",
    "E5a mechanism"
   ],
   "nice": [
    "X1 Indian English at full scale"
   ],
   "should": [
    "E3 accuracy and flips with four models (with E5b) and CoQA retention",
    "E4 realistic arm (right after E3)",
    "X4 marker-protection mitigation on retention and accuracy (exploratory; method frozen in the generation pre-commitment)",
    "E6 ReDial"
   ]
  },
  "seed": 86,
  "study": "main"
 }
}
```
