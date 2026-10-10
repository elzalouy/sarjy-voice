# Speech recognition benchmark

| Field | Value |
| --- | --- |
| Corpus | ar-sa 1.0.0 |
| Fingerprint | 7a7b374b7817a48782741035f473967acb7e20b7c1cd074c1e5596adbdac92b0 |
| Items | 37 (ar-sa-001 through ar-sa-037) |
| Item order | Same order as load_corpus() |
| STT model | nova-3 |
| STT locale | ar-SA |
| Speaker | eleven_flash_v2_5 |
| Speaker voice | pNGUEZc2WLum9bjP611g |
| Audio | ElevenLabs eleven_flash_v2_5 voice pNGUEZc2WLum9bjP611g spoke each tts_text. Phones, prices, dates, and times are spelled before synthesis. The audio is not stored and is not user audio. A digit writing in the corpus is still accepted. |

## Quality

| Measure | Value |
| --- | --- |
| Accepted | 2 / 37 (5.4%) |
| Mean word error rate | 0.383 |

Accepted means the hypothesis equals the expected transcript or one
accepted writing, after trimming and collapsing spaces. Word error
rate ignores punctuation and Arabic vowel marks. A word error rate
of 0 can still be rejected when only punctuation or vowel marks differ.

## Latency

Milliseconds from the last speech sample sent to the streaming recognizer until the last final transcript. Trailing silence used to end the utterance is not part of the speech. If no final transcript arrives, latency is not available.

| Measure | Value |
| --- | --- |
| Items with a measurement | 37 |
| Median | 377 ms |
| P95 | 1898 ms |
| Minimum | 184 ms |
| Maximum | 12166 ms |

The P95 uses the nearest rank of these 37 items. It describes this run only.

## Names and places

5 of 5 missed.

| Id | Kind | WER | Expected | Hypothesis |
| --- | --- | --- | --- | --- |
| ar-sa-011 | name_place | 0.25 | أنا محمد، وأبي محطة في حي النرجس بالرياض | أنا محمد و أبي محطة في حي النرجس بالرياض. |
| ar-sa-012 | name_place | 0.88 | اسمي نورة عبدالله، وأسكن في جدة حي الشاطئ | أسمي نوره عبد الله أسكن في جد حي الشاط الله |
| ar-sa-013 | name_place | 0.14 | وديني أقرب محطة على طريق الملك فهد | ودني أقرب محطة على طريق الملك فهد |
| ar-sa-014 | name_place | 0.00 | الموعد عند غادة في الدمام، حي الفيصلية | الموعد عند غادة في الدمام حي الفيصلية. |
| ar-sa-015 | name_place | 0.00 | فيه محطة قريبة من الحرم في مكة؟ | فيه محطة قريبة من الحرم في مكة |

## Numbers

8 of 8 missed.

| Kind | Missed |
| --- | --- |
| phone | 2 of 2 |
| price | 2 of 2 |
| date | 2 of 2 |
| time | 2 of 2 |

| Id | Kind | WER | Expected | Hypothesis |
| --- | --- | --- | --- | --- |
| ar-sa-016 | phone | 0.82 | رقمي صفر خمسة خمسة صفر صفر صفر صفر صفر صفر واحد | رقمي صفر55000 واحد |
| ar-sa-017 | phone | 0.73 | كلمني على زائد تسعة ستة ستة خمسة صفر صفر صفر صفر صفر صفر صفر اثنين | كلمني على زائد 966 ، 5 صفر 2 |
| ar-sa-018 | price | 0.33 | الشحن بـ خمسة عشر ريال ونص | الشحن بخمسة عشر ريال ونص. |
| ar-sa-019 | price | 0.71 | الباقة بـ تسعة وأربعين ريال وتسعين هللة | الباقة بـ ٩٤٠ ريال و تسعين: هل |
| ar-sa-020 | date | 0.17 | الموعد يوم الجمعة تسعة أكتوبر ألفين وستة وعشرين | الموعد يوم الجمعة 9 اكتوبر 2026 |
| ar-sa-021 | date | 0.40 | الإجازة تبدأ خمسة عشر رمضان | الجازة تبدأ خمسة ؟ رمضان. |
| ar-sa-022 | time | 0.40 | تعال الساعة ثلاثة ونص العصر | تعال الساعة ثلاثة و نصف العصر. |
| ar-sa-023 | time | 0.25 | الاجتماع الساعة ثمانية الصبح | الاجتماع الساعة ثمانية الصبح. الاجتماع |

## Mixed language

5 of 5 missed.

| Id | Kind | WER | Expected | Hypothesis |
| --- | --- | --- | --- | --- |
| ar-sa-024 | mixed_language | 0.20 | أبي أطلب iPhone من التطبيق | أبي أطلب أيفوني من التطبيق. |
| ar-sa-025 | mixed_language | 0.50 | الشاحن يدعم Fast Charge ولا لا؟ | الشاحن يدعم فاستشارج أم لا ؟ |
| ar-sa-026 | mixed_language | 0.62 | الـ GPS ما يشتغل، أعد تشغيل الـ Bluetooth | الجيبيس ما يشتغل إعادة تشغيل البلوتوث. |
| ar-sa-027 | mixed_language | 0.83 | خلصت الـ battery، أحتاج Power Bank | .خلصت البتري ؟ احتاج باور بانك |
| ar-sa-028 | mixed_language | 0.62 | احجز لي موعد في الـ Calendar يوم الأحد | احجز لموعد في الكالندر يوم الاحد |

## Dialect and unclear speech

This group is items marked `unclear`, plus items whose notes say the
form is dialect.

4 of 5 missed.

| Id | Kind | WER | Expected | Hypothesis |
| --- | --- | --- | --- | --- |
| ar-sa-032 | meaning | 0.50 | الساعة وحدة | واحدة |
| ar-sa-034 | unclear | 1.00 | آآ وين ال | آون آون |
| ar-sa-035 | unclear | 1.00 | همم | هم |
| ar-sa-037 | unclear | 0.50 | خمسطعش دقيقة | 15 دقيقة |

## Other misses

13 of 14 missed.

| Id | Kind | WER | Expected | Hypothesis |
| --- | --- | --- | --- | --- |
| ar-sa-001 | greeting | 0.50 | السلام عليكم | عليكم |
| ar-sa-002 | greeting | 0.00 | هلا والله، كيفك؟ | هلا والله ، كيفك ؟ |
| ar-sa-003 | greeting | 0.00 | صباح الخير، وش أخبارك؟ | صباح الخير ، وش أخبارك ؟ |
| ar-sa-004 | greeting | 0.50 | يعطيك العافية | يعطيك العافيه. |
| ar-sa-006 | question_instruction | 0.33 | وين أقرب محطة شحن من موقعي؟ | أقرب محطة شحن من موقع ؟ |
| ar-sa-007 | question_instruction | 0.14 | أبي أشحن السيارة الحين، كم ياخذ الوقت؟ | أبي أشحن السيارة الحين كم يأخذ الوقت |
| ar-sa-008 | question_instruction | 0.50 | افتح لي المحطات القريبة، ووريني الأسعار | أفتح لي المحطات القريبة و رني الأسعار. |
| ar-sa-009 | question_instruction | 0.50 | وش حالة طلبي الأخير؟ | مش حالة الطلب الأخير. |
| ar-sa-010 | question_instruction | 0.40 | أبغى ألغي الموعد حق بكرة | أبغى أبغى الغّي الموعد حقّ بكرة. |
| ar-sa-029 | meaning | 0.00 | تمام | تمام. |
| ar-sa-030 | meaning | 0.00 | على خير | على خير. |
| ar-sa-031 | meaning | 0.25 | قابلني عند الملك فهد | قبلني عند الملك فهد |
| ar-sa-033 | meaning | 0.17 | يعني أبغى الشحن بس مو متأكد | يعني ابغى الشحن بس مو متأكد |

## Comparison

A later model or locale must use corpus ar-sa 1.0.0, fingerprint 7a7b374b7817a48782741035f473967acb7e20b7c1cd074c1e5596adbdac92b0, and the same item ids. Compared with macos-say voice Majed. Changed items: ar-sa-001, ar-sa-002, ar-sa-003, ar-sa-004, ar-sa-005, ar-sa-006, ar-sa-007, ar-sa-008, ar-sa-009, ar-sa-010, ar-sa-011, ar-sa-012, ar-sa-013, ar-sa-014, ar-sa-016, ar-sa-017, ar-sa-018, ar-sa-019, ar-sa-020, ar-sa-021, ar-sa-022, ar-sa-023, ar-sa-024, ar-sa-025, ar-sa-026, ar-sa-027, ar-sa-028, ar-sa-029, ar-sa-030, ar-sa-032, ar-sa-033, ar-sa-034, ar-sa-035.

## Every item

| Id | Category | Accepted | WER | Latency ms | Hypothesis |
| --- | --- | --- | --- | --- | --- |
| ar-sa-001 | greeting | no | 0.50 | 1898 | عليكم |
| ar-sa-002 | greeting | no | 0.00 | 401 | هلا والله ، كيفك ؟ |
| ar-sa-003 | greeting | no | 0.00 | 725 | صباح الخير ، وش أخبارك ؟ |
| ar-sa-004 | greeting | no | 0.50 | 505 | يعطيك العافيه. |
| ar-sa-005 | greeting | yes | 0.00 | 515 | تصبح على خير |
| ar-sa-006 | question_instruction | no | 0.33 | 12166 | أقرب محطة شحن من موقع ؟ |
| ar-sa-007 | question_instruction | no | 0.14 | 299 | أبي أشحن السيارة الحين كم يأخذ الوقت |
| ar-sa-008 | question_instruction | no | 0.50 | 352 | أفتح لي المحطات القريبة و رني الأسعار. |
| ar-sa-009 | question_instruction | no | 0.50 | 373 | مش حالة الطلب الأخير. |
| ar-sa-010 | question_instruction | no | 0.40 | 261 | أبغى أبغى الغّي الموعد حقّ بكرة. |
| ar-sa-011 | name_place | no | 0.25 | 211 | أنا محمد و أبي محطة في حي النرجس بالرياض. |
| ar-sa-012 | name_place | no | 0.88 | 1518 | أسمي نوره عبد الله أسكن في جد حي الشاط الله |
| ar-sa-013 | name_place | no | 0.14 | 340 | ودني أقرب محطة على طريق الملك فهد |
| ar-sa-014 | name_place | no | 0.00 | 217 | الموعد عند غادة في الدمام حي الفيصلية. |
| ar-sa-015 | name_place | no | 0.00 | 438 | فيه محطة قريبة من الحرم في مكة |
| ar-sa-016 | number | no | 0.82 | 204 | رقمي صفر55000 واحد |
| ar-sa-017 | number | no | 0.73 | 201 | كلمني على زائد 966 ، 5 صفر 2 |
| ar-sa-018 | number | no | 0.33 | 275 | الشحن بخمسة عشر ريال ونص. |
| ar-sa-019 | number | no | 0.71 | 188 | الباقة بـ ٩٤٠ ريال و تسعين: هل |
| ar-sa-020 | number | no | 0.17 | 383 | الموعد يوم الجمعة 9 اكتوبر 2026 |
| ar-sa-021 | number | no | 0.40 | 227 | الجازة تبدأ خمسة ؟ رمضان. |
| ar-sa-022 | number | no | 0.40 | 208 | تعال الساعة ثلاثة و نصف العصر. |
| ar-sa-023 | number | no | 0.25 | 1243 | الاجتماع الساعة ثمانية الصبح. الاجتماع |
| ar-sa-024 | mixed_language | no | 0.20 | 425 | أبي أطلب أيفوني من التطبيق. |
| ar-sa-025 | mixed_language | no | 0.50 | 184 | الشاحن يدعم فاستشارج أم لا ؟ |
| ar-sa-026 | mixed_language | no | 0.62 | 377 | الجيبيس ما يشتغل إعادة تشغيل البلوتوث. |
| ar-sa-027 | mixed_language | no | 0.83 | 448 | .خلصت البتري ؟ احتاج باور بانك |
| ar-sa-028 | mixed_language | no | 0.62 | 384 | احجز لموعد في الكالندر يوم الاحد |
| ar-sa-029 | ambiguous | no | 0.00 | 582 | تمام. |
| ar-sa-030 | ambiguous | no | 0.00 | 519 | على خير. |
| ar-sa-031 | ambiguous | no | 0.25 | 351 | قبلني عند الملك فهد |
| ar-sa-032 | ambiguous | no | 0.50 | 396 | واحدة |
| ar-sa-033 | ambiguous | no | 0.17 | 272 | يعني ابغى الشحن بس مو متأكد |
| ar-sa-034 | ambiguous | no | 1.00 | 418 | آون آون |
| ar-sa-035 | ambiguous | no | 1.00 | 363 | هم |
| ar-sa-036 | ambiguous | yes | 0.00 | 493 | شحن |
| ar-sa-037 | ambiguous | no | 0.50 | 307 | 15 دقيقة |
