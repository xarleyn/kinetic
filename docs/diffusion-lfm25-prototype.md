# CPU-прототип masked diffusion для LFM2 MoE

После исходной Python-сверки добавлен
[CPU prefix reuse: побитовая сверка и замеры](diffusion-lfm25-prefix-reuse.md).
Полный пересчёт, описанный ниже, остаётся эталоном и режимом по умолчанию;
оптимизация включается флагом `--prefix-reuse` в новой сборке.

`kinetic diffuse` исполняет отдельный экспериментальный режим. Он проверен
на уменьшенной FP32-модели с настоящими блоками Transformers LFM2 MoE:
40 слоёв, из них 30 ShortConv и 10 attention, 64 эксперта, top-4,
два начальных dense FFN, GQA 4/2. Уменьшены width до 32, словарь до 384,
dense FFN до 184, expert FFN до 24. Веса и mask embedding синтетические,
seed `20261112`; модель не обучена и не предназначена для ответов на вопросы.
Получилось 5 888 000 параметров, GGUF занимает 23 597 184 байта (22,5 MiB).
Полный checkpoint summerAIO 24B в этой проверке не использовался.

## Контракт исполнения

- Только CPU, batch 1. Canvas 1–64, шагов 1–16, новых токенов 1–256.
- Непустой исходный промпт сохраняется целиком. `prefix_window` ограничивает
  только добавляемую историю генерации. Промпт плюс окно — не более 2048 токенов.
- На каждом шаге полностью пересчитываются префикс и canvas, позиции RoPE
  начинаются с нуля. KV и ShortConv state между проходами не сохраняются.
- Prefix-LM attention: запрос `q < prefix_len` видит `k <= q`, запрос canvas
  видит весь префикс и весь canvas. Свёртки остаются каузальными.
- В отмеченных позициях embedding заменяется вектором из safetensors
  `mask_embedding`, F32 `[hidden_size]`. Отдельный mask token не нужен.
- Fixed greedy sampler: temperature 0, pad запрещён, confidence — log-softmax
  выбранного токена. После шага `s` остаётся
  `canvas_len * (steps - s - 1) // steps` масок. Последний шаг заполняет остаток.
- Коммиты одновременны. Равные confidence упорядочиваются по возрастанию
  позиции, равные логиты — по возрастанию ID токена. У `torch.topk` автора нет
  переносимого правила для равных confidence; harness дополнительно проверяет
  итоговые токены и число проходов неизменённым `generate()` автора.
- EOS прекращает проходы, когда он закоммичен и все предыдущие позиции
  заполнены. Результат блока обрезается перед первым EOS.
- Repetition penalty, remask, stochastic sampling и prefix reuse не включены.

Веса маршрутизатора diffusion-пути нормализуются по формуле закреплённого
Transformers: `selected_weights / (sum(selected_weights) + 1e-6)`.
Обычный GGUF autoregressive-путь сохраняет своё правило нормализации с floor.

GGUF должен явно содержать строку
`kinetic.diffusion.mode = "lfm2-prefix-lm-v1"` и архитектуру `lfm2moe`.
Обычный загрузчик decoder отказывает таким файлам с указанием `kinetic diffuse`.
Diffusion-загрузчик отказывает немаркированным файлам и другим архитектурам.
Поддержан существующий набор блоков LFM2 MoE без sliding window, MTP,
per-layer inputs и shared KV.

## Запуск на этой машине

Сборка CPU-прототипа:

```powershell
cargo build --locked --release -p kinetic-cli --no-default-features
```

Готовый бинарь: `D:\LLM\kinetic-diffusion-cpu.exe`.
Модель: `D:\LLM\models\lfm25-diffusion-cpu-fixture`.
Основной `D:\LLM\kinetic.cmd` продолжает использовать ранее проверенную CUDA-сборку.

Пример с токенами из сверочного набора:

```powershell
& D:\LLM\kinetic-diffusion-cpu.exe diffuse `
  D:\LLM\models\lfm25-diffusion-cpu-fixture\lfm2moe-diffusion-fixture-f32.gguf `
  --mask-embedding D:\LLM\models\lfm25-diffusion-cpu-fixture\mask.safetensors `
  --prompt-tokens D:\LLM\logs\kinetic-diffusion-cpu-20261112\case-2-c16-s4\prompt.json `
  --canvas 16 --steps 4 --max-new-tokens 33 --prefix-window 8 -t 6
```

Вместо JSON токенов можно передать `--prompt "text"`; применяется сырой
текст без chat template. `--dry-run` проверяет файлы, маску и настройки,
печатает план и не выполняет forward. `--trace-dir` принимает новую или
пустую директорию, чтобы не перезаписывать предыдущую сверку.

## Воспроизведение Python-сверки

Harness `scripts/diffusion_reference.py` использует настоящий
`Lfm2MoeForCausalLM` на CPU, экспортирует его FP32-веса в GGUF и вызывает
локальный Kinetic. Python нужен для ML-референса, не для автоматизации
установки или загрузки. Новые зависимости не устанавливались.

Закреплён исходник
[summerAIO/modeling_diffusion_lfm.py](https://huggingface.co/summerAIO/diffusion-lfm2.5-24b-a2b/blob/f0f185c3b8b4da394527e9efbe9a0c87c94b5c01/modeling_diffusion_lfm.py),
ревизия `f0f185c3b8b4da394527e9efbe9a0c87c94b5c01`, SHA-256
`bf3e7124106a6dbd92900f908ba1d815cb8084ad73a55bef394b1232e410a8dd`.
Harness проверяет SHA перед импортом локального файла; сетевой remote code
и загрузка полного checkpoint не используются. `use_prefix_cache=False`.

Проверенное окружение: PyTorch `2.14.1+cu126`, Transformers `5.20.2`,
NumPy `2.2.6`; CPU FP32, один поток PyTorch и шесть потоков Kinetic.
Пути `--model-dir` и `--output` должны быть новыми или пустыми.

```powershell
& D:\LLM\ai-nov\env\idol\Scripts\python.exe scripts\diffusion_reference.py `
  --kinetic D:\LLM\kinetic-diffusion-cpu.exe `
  --source-dir D:\LLM\models\diffusion-lfm2.5-24b-a2b-reference `
  --gguf-python D:\LLM\llama.cpp\gguf-py `
  --model-dir D:\LLM\models\lfm25-diffusion-cpu-fixture-recheck `
  --output D:\LLM\logs\kinetic-diffusion-cpu-recheck
```

Сверяются **все** логиты canvas до запрета pad, ID кандидатов,
confidence маскированных позиций, наборы коммитов, canvas до/после,
полный префикс, число проходов и окончательные ID. Допуск логитов и confidence:
`abs(actual - reference) <= 5e-5 + 2e-4 * abs(reference)`.
Дискретные результаты должны совпадать точно.

| Canvas | Шагов на блок | Лимит генерации | Проходов | Сравнено логитов | Max abs error |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 3 | 12 | 4 608 | 5.13e-6 |
| 16 | 1 | 16 | 1 | 6 144 | 2.98e-6 |
| 16 | 4 | 33 | 12 | 50 688 | 7.15e-6 |
| 32 | 3 | 39 | 6 | 44 928 | 4.89e-6 |
| 64 | 4 | 71 | 8 | 109 056 | 6.24e-6 |

Итого 5 сценариев, 39 проходов, 215 424 логита. Включены короткий последний
canvas, несколько блоков, шагов больше позиций и промпт длиннее окна истории.
Отчёты, хеши бинаря/модели/маски/референса, версии и сырые дампы находятся
в `D:\LLM\logs\kinetic-diffusion-cpu-20261112` (`run.json`, `results.json`).
`hf/` рядом с GGUF содержит исходную уменьшенную модель Transformers.

## Найденные расхождения и границы проверки

Первый прогон canvas 64 выявил почти равные оценки четвёртого и пятого
экспертов на слое 21: разница около `6e-8`. После расхождения выбора
эксперта ошибка логитов достигала `0.763`. Послойные дампы локализовали
расхождение; затем diffusion-путь получил правильный знаменатель `sum + 1e-6`.
Повторная сверка того же seed, весов и промптов прошла с исходными допусками.
Первый провал сохранён в `D:\LLM\logs\kinetic-diffusion-20261112`, диагностика —
в `D:\LLM\logs\kinetic-diffusion-diag`. Это также пример, почему близкие
маршруты MoE нельзя оценивать только по итоговому top-1 или широкому допуску.
Другие веса, платформы и численные режимы требуют собственной сверки:
побитовая эквивалентность с PyTorch и устойчивость маршрутов на всех входах
этим экспериментом не установлены.

Rust unit-тесты проверяют расписание, tie break, EOS с незаполненными
предыдущими позициями, короткий последний блок и нечисловые логиты.
Интеграционные проверки на GGUF подтверждают каузальность префикса,
влияние будущего canvas, замену masked embedding, отсутствие состояния
между проходами и отказ для неверных размеров/ID/маски/обычного decoder.

```powershell
$env:KINETIC_TEST_DIFFUSION_GGUF='D:\LLM\models\lfm25-diffusion-cpu-fixture\lfm2moe-diffusion-fixture-f32.gguf'
$env:RAYON_NUM_THREADS='6'
cargo test --locked --release -p kinetic-models --test diffusion -- --ignored
```

На итоговом исходном коде также прошли `cargo fmt --all -- --check`,
`git diff --check`, Clippy для CLI со всеми features и `-D warnings`,
308 библиотечных тестов workspace (4 внешних теста пропущены), 44 теста CLI
и 2 указанных интеграционных теста. Для Windows CUDA-проверок использован `CUDAHOSTCXX`
из MSVC Build Tools 14.44.35207. Полный пример команды есть в инструкции сборки.
CLI дополнительно проверен на `--help`, `--dry-run`, отказ для canvas 0 и
запрет обычного `run` с маркированным diffusion GGUF.

Это проверка механики и переноса вычислений, не качества diffusion-адаптации,
скорости полного 24B checkpoint, BF16/квантования, GPU, API или prefix reuse.
Проверка ускоренного пути описана отдельно в отчёте о prefix reuse выше.
