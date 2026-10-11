# Speech recognition benchmark

| Field | Value |
| --- | --- |
| Corpus | ar-sa 2.0.0 |
| Fingerprint | 1267eb0a8565be6684f818f9f0e590cab9cd007263feaa8696db1f783d28cdaa |
| Items | 5 (000098_2-0010 through 000098_2-0014) |
| Item order | Same order as load_corpus() |
| STT model | nova-3 |
| STT locale | ar-SA |
| Speaker | recording |
| Speaker voice | wav |
| Audio | 5 wav files from /Volumes/work/sarj/sarjy-voice/src/sarjy_agent/eval/data. The txt file with the same name is the expected transcript. A digit writing in the corpus is still accepted. |

## Quality

| Measure | Value |
| --- | --- |
| Accepted | 0 / 5 (0.0%) |
| Mean word error rate | 0.441 |

Accepted means the hypothesis equals the expected transcript or one
accepted writing, after trimming and collapsing spaces. Word error
rate ignores punctuation and Arabic vowel marks. A word error rate
of 0 can still be rejected when only punctuation or vowel marks differ.

## Latency

Milliseconds from the last speech sample sent to the streaming recognizer until the last final transcript. Trailing silence used to end the utterance is not part of the speech. If no final transcript arrives, latency is not available.

| Measure | Value |
| --- | --- |
| Items with a measurement | 5 |
| Median | 358 ms |
| P95 | 601 ms |
| Minimum | 292 ms |
| Maximum | 601 ms |

The P95 uses the nearest rank of these 5 items. It describes this run only.

## Names and places

0 of 0 missed. No misses.

## Numbers

0 of 0 missed.

| Kind | Missed |
| --- | --- |
| phone | 0 of 0 |
| price | 0 of 0 |
| date | 0 of 0 |
| time | 0 of 0 |

No misses.

## Mixed language

0 of 0 missed. No misses.

## Dialect and unclear speech

This group is items marked `unclear`, plus items whose notes say the
form is dialect.

0 of 0 missed. No misses.

## Other misses

5 of 5 missed.

| Id | Kind | WER | Expected | Hypothesis |
| --- | --- | --- | --- | --- |
| 000098_2-0010 | recorded | 0.53 | اليهودية، والمسحية، والهندوسية اللي ما عمرنا شفنا حد منهم ينتقدها او يتكلم عنها اصلا، الرمي كله للأسلام شفتوا الدليل، | اليهوديه والمسيحيه والهندوسيه اللي مع عمرنا شفنا احد منهم ينتقدها ويتكلم عنها اصلا الرامي كله للإسلام شفتو الدليل |
| 000098_2-0011 | recorded | 0.53 | يعني شفتوه انتم وسمعتوه، المذيعة تقول بما في ذلك النص القرآني رمي للأسلام، راحت تقول له طبعا، | شفتوه أنتم وسمعتوه المذيع تقول في ذلك النص القراني الرمي للاسلام تقول طبعا |
| 000098_2-0012 | recorded | 0.50 | طبعا، احكي شلون تغى تجدد الخطاب الديني؟ وش، تبغى تجدد الخطاب الديني، تبي تغير نصوص القرآن، | طبع طبع شلون تبغى تجدد الخطاب الديني تغير نصوص القرآن |
| 000098_2-0013 | recorded | 0.25 | اكتشفت شئ يعني والعياذ بالله ما كان يعرفه النبي عليه الصلاة والسلام، | اكتشفت شي والعياذ بالله ما كان يعرفه النبي عليه الصلاه والسلام |
| 000098_2-0014 | recorded | 0.40 | ما ابغى اقول اكثر من كذا صراحة ما اقدر حتى أقولها كا كمثال وإلا كنكتة، | ما ابغى اقول اكبر من كذا بصراحه ما اقدر حتى اقولها كمثال ولا نكته |

## Comparison

A later model or locale must use corpus ar-sa 2.0.0, fingerprint 1267eb0a8565be6684f818f9f0e590cab9cd007263feaa8696db1f783d28cdaa, and the same item ids. No earlier result was loaded. This run is the baseline.

## Every item

| Id | Category | Accepted | WER | Latency ms | Hypothesis |
| --- | --- | --- | --- | --- | --- |
| 000098_2-0010 | recorded | no | 0.53 | 358 | اليهوديه والمسيحيه والهندوسيه اللي مع عمرنا شفنا احد منهم ينتقدها ويتكلم عنها اصلا الرامي كله للإسلام شفتو الدليل |
| 000098_2-0011 | recorded | no | 0.53 | 292 | شفتوه أنتم وسمعتوه المذيع تقول في ذلك النص القراني الرمي للاسلام تقول طبعا |
| 000098_2-0012 | recorded | no | 0.50 | 432 | طبع طبع شلون تبغى تجدد الخطاب الديني تغير نصوص القرآن |
| 000098_2-0013 | recorded | no | 0.25 | 601 | اكتشفت شي والعياذ بالله ما كان يعرفه النبي عليه الصلاه والسلام |
| 000098_2-0014 | recorded | no | 0.40 | 330 | ما ابغى اقول اكبر من كذا بصراحه ما اقدر حتى اقولها كمثال ولا نكته |
