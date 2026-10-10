# Text-to-speech comparison

| Field | Flash v2.5 | Multilingual v2 |
| --- | --- | --- |
| Model | eleven_flash_v2_5 | eleven_multilingual_v2 |
| Voice id | pNGUEZc2WLum9bjP611g | pNGUEZc2WLum9bjP611g |
| Corpus | ar-sa 1.0.0 | same |
| Fingerprint | 7a7b374b7817a48782741035f473967acb7e20b7c1cd074c1e5596adbdac92b0 | same |
| Items | 37 | 37 |
| Hearing check | nova-3 ar-SA | same |

## What was measured

Both runs speak the same corpus sentences, in the same order, with the same voice. Numbers are spelled before synthesis, and provider text normalization is off for both models. The voice is the configured voice. Both models accepted it, so only the model changes. The API key cannot list the voice library, so no other voice was tested.

Latency is the time from the request to the first audio frame. The hearing check is Nova-3 listening to that audio. A sentence is accepted when the transcript matches the expected spoken text or an accepted writing. This is not a human judgment of naturalness or of a Saudi accent. No voice is called the best Saudi voice.

## Latency

| Measure | Flash v2.5 | Multilingual v2 |
| --- | --- | --- |
| Items with a first frame | 37 | 37 |
| Median | 189 ms | 945 ms |
| P95 | 338 ms | 1233 ms |
| Minimum | 163 ms | 819 ms |
| Maximum | 744 ms | 1741 ms |

The P95 uses the nearest rank of the measured items in that run.

## Hearing check

| Measure | Flash v2.5 | Multilingual v2 |
| --- | --- | --- |
| Accepted sentences | 4 / 37 | 3 / 37 |
| Accepted number sentences | 1 / 8 | 0 / 8 |
| Mean word error rate | 0.462 | 0.455 |

## Number sentences

| Id | Kind | Flash transcript | Multilingual transcript |
| --- | --- | --- | --- |
| ar-sa-016 | phone | رقم 0.5.5. | رقم 05, 5, |
| ar-sa-017 | phone | كلمني على زائد تسعة ستة. 65000 اثنين | كلمني على زائد 9.6 50002002 |
| ar-sa-018 | price | الشحن بخمسة عشر ريال ونص. الشحن | الشحن بخمسة عشر ونص الشحن بخمسة |
| ar-sa-019 | price | الباقة بالتسعة هو ريال وتسعين: " هل... | الباقة بتسعة و أربع ٤٠ ريال وتسعين هلة. |
| ar-sa-020 | date | الموعد يوم الجمعة 9 أكتوبر 2026 | الموعد يوم الجمعة 9 اكتوبر 2026 |
| ar-sa-021 | date | تبدأ خمستعش رمضان الاجازة | الإجازة تبدأ ٥ رمضان. |
| ar-sa-022 | time | تعال الساعة ثلاثة ونُصم تعال الساعة العصر | تعال الساعة ثلاث و نص ، العصر. |
| ar-sa-023 | time | ولأجتماع الساعة 8. الصبح الاجتماع | الاجتماع الساعة ثمانية الصبح. |

## Decision

Primary model: `eleven_flash_v2_5`.

The measured pronunciation and latency do not justify a change. Flash v2.5 stays primary. Accent and naturalness were not judged by a listener.
