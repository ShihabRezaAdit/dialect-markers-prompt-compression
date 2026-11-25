# P0 pre-commitment (Idea 86)

Written 2026-11-25T08:19:08Z at git commit `01d59b982b81fec12dfcf3040062c1d1e094b84d`. Thresholds, rules, samples and seeds below do not change after this file is committed; any change is logged with its reason in the decisions log and reported in the paper.

Rendering, compression and generation of pilot items verify the block below and refuse to start if anything differs.

```json
{
 "meta": {
  "git_commit": "01d59b982b81fec12dfcf3040062c1d1e094b84d",
  "unit_tests": {
   "code_hash": "128df29d0c66610d4d6f4ad4a1f4e2b875f657a678abab89a1400c65e8227477",
   "git_commit": "01d59b982b81fec12dfcf3040062c1d1e094b84d",
   "passed": true,
   "utc": "2026-11-25T08:09:33Z"
  },
  "written_utc": "2026-11-25T08:19:08Z"
 },
 "state": {
  "code_hash": "128df29d0c66610d4d6f4ad4a1f4e2b875f657a678abab89a1400c65e8227477",
  "config_hash": "308b10e4ae5fb0e513b0ce6e143694503370a25337c4f05dc1dd4712c83346eb",
  "decision_rule": {
   "Go": "S1 and S2 met; E1 to E6 in full; the paper leads with RQ1 and RQ2",
   "Go, realistic-pipeline focus": "S2 met, S1 not; E1 to E6 in full; the paper leads with RQ2",
   "Go, retention focus": "S1 met, S2 not; E1 to E6 in full; the paper leads with RQ1 and RQ3",
   "No-go": "neither met; stop and move to the backup idea, or an equivalence short paper if the advisor prefers",
   "applies_when": "G1 to G5 pass, or a gate that still fails was decided under the gate-fix rule (fixes only for bug, environment or input problems apart from the gates-table actions; each logged with its cause before the re-check; at most 2 extra days; then the project lead decides, logs the reason and reports it to the advisor)",
   "status": "recommendation; the decision is taken with the advisor"
  },
  "env_locks_sha256": {
   "env-compress": "b1f62ac32a9560096bf82f1c386226b74fb988d32106b3bdc7e185f971dc7eb1",
   "env-gen": "dfadda9dc108fc267c8f2162e5cff2d8dc38ed2dd8fb2bc5e18474c9ae429630",
   "env-render": "a68d547098c4dfcf8607bbaa2cb8e652cba6bf8ab2e494759039317b1d21ccb9",
   "env-sc": "15a188f9feaec676bc74447c7b92363a72e33372d28551966216bf94beecd052",
   "env-stats": "0f02f72148a7d518d56c835570f55f1aa2a7601141f16ca224206aced9a16da0"
  },
  "experiment": "P0",
  "gates": {
   "g1_compress_texts_per_compressor": 20,
   "g1_generate_min_same": 196,
   "g1_generate_prompts_per_model": 200,
   "g1_render_pairs_per_dialect": 20,
   "g2_band": [
    0.35,
    0.65
   ],
   "g2_max_diff": 0.05,
   "g3_min_overlap": 0.9,
   "g4_max_abs": 0.03,
   "g5_min_answer_preserved": 0.6,
   "g5_min_content_yes": 43
  },
  "item_lists_sha256": {
   "audit_pairs": "8c72fc39d351dd86c5e35caae01d43466ea53f86aafde625fa2b2af359122c8f",
   "paragraph_numbers": "5ab05a180f472a8f65154e297310fcc2e578de2aa63cc023312075aac35ae5f5",
   "question_ids": "f6e8d95fa000c8052c34ca80b5eadfddb5d79619641e537c18920f585515a5aa",
   "tweet_rows": "f13a7414ffc7cfa0c7176268055a605f7cbedb0c653a8e45a8e59b12381b8b26"
  },
  "model_revisions": {
   "M1": {
    "repo": "meta-llama/Llama-3.1-8B-Instruct",
    "revision": "0e9e39f249a16976918f6564b8830bc894c89659"
   },
   "M2": {
    "repo": "Qwen/Qwen2.5-14B-Instruct",
    "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
   },
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
   "internal_plan_tex": "7d24f964391bc55c0eb45b0e8a4924e6c631a2d5648f7109dd4411c6aea8ccae",
   "pilot_plan_tex": "1cc29e55eea221703622db453d093facd01e0ad031f783b2789d2b79714b129c"
  },
  "seed": 86,
  "signals": {
   "s1_alpha": 0.05,
   "s1_cells": 6,
   "s1_floor": 0.04,
   "s2_alpha": 0.05,
   "s2_cells": 2,
   "s2_floor": 0.03
  },
  "squad": {
   "config": "plain_text",
   "repo": "rajpurkar/squad",
   "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
   "split": "train"
  },
  "tweets": {
   "bytes": 8451854,
   "file": "real_aae/real_aae.csv",
   "rows": 17479,
   "sha256": "7dba90985790315ede8fe9d5494408ce8355d74437d5e98cea2789fafc09e5bc",
   "source": "REAL-AAE"
  }
 }
}
```
