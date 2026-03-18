# Sermon Failure Cases

Generated from the optional ChromaDB + RAGAS smoke run on March 18, 2026 after cleaning the default staged sermon corpus with [`data/eval/sermon_excluded_files.txt`](/Users/yinshi/Documents/breadrag/data/eval/sermon_excluded_files.txt):

```bash
python scripts/run_eval.py --config configs/sermon_chromadb_ragas.yaml
```

Run summary on 28 labeled sermon questions:

| config | Recall@3 | MRR | Hit Rate | RAGAS Context Recall |
|---|---:|---:|---:|---:|
| `dense_sentence_top3_sermon_chromadb_ragas` | 0.7857 | 0.6429 | 0.7857 | 0.7857 |

Main patterns from this smoke run:

- The multilingual dense retriever is clearly usable on the Chinese sermon path, and it beats the earlier BM25 sermon baseline by a large margin.
- The misses are still mostly bridge or scripture-reference questions where many sermons share similar church vocabulary.
- The optional `ragas_context_recall` lines up closely with `Recall@3` here because both are using gold document ids rather than generated answers.
- Excluding the two confirmed duplicate sermon files made the remaining miss set cleaner, but the ChromaDB smoke run still misses a few true retrieval cases.

## Case 1: The retriever stayed near doctrine language but missed the suffering sermon

Question:

> 腓立比书一章二十九节说蒙恩的人除了信基督还要经历什么

Gold doc:

- `[初信者话语] 第七讲_基督徒的苦难 (崔丰宇P)_原文`

Observed top 3:

1. `[初信者话语] 第三讲_圣灵的内住和引导 (崔丰宇P)_原文`
2. `[圣经讲座] 第四篇_ 金松奎 P_2022 福音布道会_Day4_原文`
3. `[圣经讲座] 第三篇_ 金松奎 P_2022 福音布道会r_Day3_原文`

Why this failed:

- The query is anchored on a Bible reference plus broad faith vocabulary, so several sermons sound semantically similar.
- The target sermon is about suffering, but the dense model seems to over-weight nearby doctrinal language.
- This is a good sign that scripture-reference questions need sharper wording or more labels.

## Case 2: The final-day sermon was confused with other salvation-heavy sermons

Question:

> 最后一天布道会说耶稣基督从死里复活使信徒有什么

Gold doc:

- `[圣经讲座] 第六篇_金松奎P_2022 福音布道会_Day6_原文`

Observed top 3:

1. `[初信者话语] 第一讲_得救的确信(崔丰宇P)_原文 (1)`
2. `[初信者话语] 第一讲_得救的确信(崔丰宇P)_原文 (1)`
3. `[圣经讲座] 第五篇_ 金松奎 P_2022 福音布道会_Day5_原文`

Why this failed:

- The answer phrase "活泼的盼望" lives in a salvation-heavy semantic neighborhood that overlaps with other gospel sermons.
- The duplicate top hits also suggest chunk-level retrieval is pulling multiple nearby chunks from the same wrong sermon.
- This is a strong candidate for later deduping by document at ranking time.

## Case 3: Church service wording blurred together across the early believer lessons

Question:

> 哪一篇讲道开头呼吁大家在教会里面找到自己的一个侍奉的位置

Gold doc:

- `[初信者话语] 第二讲_救恩的三个阶段 (崔丰宇P)_原文`

Observed top 3:

1. `[初信者话语] 第四讲_基督徒的教会观 (崔丰宇P)_原文`
2. `[初信者话语] 第三讲_圣灵的内住和引导 (崔丰宇P)_原文`
3. `[初信者话语] 第四讲_基督徒的教会观 (崔丰宇P)_原文`

Why this still failed after cleanup:

- Excluding the duplicated `第5讲` file helped remove one noisy false positive, but the query still lands in the same church-life neighborhood.
- This question depends on a specific opening announcement rather than the central sermon topic.
- It is still a valid retrieval target, but it is noisier than the doctrine-title questions.

## Case 4: A tax collector question landed near the right seminar day but not the right one

Question:

> 第五天布道会路加福音十八章的比喻里谁比法利赛人更算为义

Gold doc:

- `[圣经讲座] 第五篇_ 金松奎 P_2022 福音布道会_Day5_原文`

Observed top 3:

1. `[初信者话语] 第二讲_救恩的三个阶段 (崔丰宇P)_原文`
2. `[初信者话语] 第二讲_救恩的三个阶段 (崔丰宇P)_原文`
3. `[圣经讲座] 第四篇_ 金松奎 P_2022 福音布道会_Day4_原文`

Why this failed:

- The query mixes a scripture reference, a role name, and seminar-day context.
- The model partially kept the seminar neighborhood, but it did not lock onto the correct day.
- This is another hint that sermon-series metadata could help retrieval later.

## Main Takeaways

- The optional ChromaDB + RAGAS run is real and useful, not just a checkbox path.
- The hardest sermon misses are not random. They cluster around repeated doctrine language, Bible references, and series-level overlap.
- The duplicate-file cleanup was worth doing first, because it removed a fake source of error and improved the stronger dense sermon runs.
- Another improvement is to keep expanding the label set with more discriminative question wording, especially for scripture-reference prompts.
