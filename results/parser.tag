Set the max num of threads to 4
Set the seed for generating random numbers to 1
Set the device with ID 0 visible
Preprocess the data
Vocab(
  num of words: 401153
  num of tags: 48
  num of rels: 46
)
Load the dataset
  size of trainset: 39832
  size of devset: 1700
  size of testset: 2416
Create the model
  n_words: 21679
  n_embed: 100
  n_tags: 48
  n_tag_embed: 100
  embed_dropout: 0.33
  n_lstm_hidden: 400
  n_lstm_layers: 3
  lstm_dropout: 0.33
  n_mlp_arc: 500
  n_mlp_rel: 100
  mlp_dropout: 0.33
  n_rels: 46
  pad_index: 0
  unk_index: 1
BiaffineParser(
  (pretrained): Embedding(401153, 100)
  (embed): Embedding(21679, 100)
  (tag_embed): Embedding(48, 100)
  (embed_dropout): IndependentDropout(p=0.33)
  (lstm): LSTM(
    (f_cells): ModuleList(
      (0): LSTMCell(200, 400)
      (1): LSTMCell(800, 400)
      (2): LSTMCell(800, 400)
    )
    (b_cells): ModuleList(
      (0): LSTMCell(200, 400)
      (1): LSTMCell(800, 400)
      (2): LSTMCell(800, 400)
    )
  )
  (lstm_dropout): SharedDropout(p=0.33, batch_first=True)
  (mlp_arc_h): MLP(
    (linear): Linear(in_features=800, out_features=500, bias=True)
    (activation): LeakyReLU(negative_slope=0.1)
    (dropout): SharedDropout(p=0.33, batch_first=True)
  )
  (mlp_arc_d): MLP(
    (linear): Linear(in_features=800, out_features=500, bias=True)
    (activation): LeakyReLU(negative_slope=0.1)
    (dropout): SharedDropout(p=0.33, batch_first=True)
  )
  (mlp_rel_h): MLP(
    (linear): Linear(in_features=800, out_features=100, bias=True)
    (activation): LeakyReLU(negative_slope=0.1)
    (dropout): SharedDropout(p=0.33, batch_first=True)
  )
  (mlp_rel_d): MLP(
    (linear): Linear(in_features=800, out_features=100, bias=True)
    (activation): LeakyReLU(negative_slope=0.1)
    (dropout): SharedDropout(p=0.33, batch_first=True)
  )
  (arc_attn): Biaffine(n_in=500, n_out=1, bias_x=True)
  (rel_attn): Biaffine(n_in=100, n_out=46, bias_x=True, bias_y=True)
)

Epoch 1 / 1000:
train: Loss: 0.6147 UAS: 87.44% LAS: 84.47%
dev:   Loss: 0.6591 UAS: 86.74% LAS: 83.25%
test:  Loss: 0.6681 UAS: 86.69% LAS: 83.56%
0:04:51.532528s elapsed

Epoch 2 / 1000:
train: Loss: 0.4338 UAS: 91.07% LAS: 88.75%
dev:   Loss: 0.4933 UAS: 90.30% LAS: 87.63%
test:  Loss: 0.4874 UAS: 90.20% LAS: 87.62%
0:04:48.197607s elapsed

Epoch 3 / 1000:
train: Loss: 0.3709 UAS: 92.34% LAS: 90.20%
dev:   Loss: 0.4230 UAS: 91.46% LAS: 89.03%
test:  Loss: 0.4263 UAS: 91.36% LAS: 88.98%
0:04:58.558112s elapsed

Epoch 4 / 1000:
train: Loss: 0.3323 UAS: 93.07% LAS: 91.02%
dev:   Loss: 0.3954 UAS: 92.24% LAS: 89.82%
test:  Loss: 0.3944 UAS: 92.05% LAS: 89.73%
0:04:47.871899s elapsed

Epoch 5 / 1000:
train: Loss: 0.3039 UAS: 93.68% LAS: 91.71%
dev:   Loss: 0.3613 UAS: 92.74% LAS: 90.47%
test:  Loss: 0.3639 UAS: 92.65% LAS: 90.42%
0:04:48.341339s elapsed

Epoch 6 / 1000:
train: Loss: 0.2907 UAS: 94.01% LAS: 92.08%
dev:   Loss: 0.3580 UAS: 93.00% LAS: 90.69%
test:  Loss: 0.3573 UAS: 93.07% LAS: 90.87%
0:04:48.365231s elapsed

Epoch 7 / 1000:
train: Loss: 0.2679 UAS: 94.41% LAS: 92.50%
dev:   Loss: 0.3382 UAS: 93.29% LAS: 90.98%
test:  Loss: 0.3391 UAS: 93.34% LAS: 91.14%
0:04:47.241473s elapsed

Epoch 8 / 1000:
train: Loss: 0.2604 UAS: 94.54% LAS: 92.68%
dev:   Loss: 0.3315 UAS: 93.42% LAS: 91.25%
test:  Loss: 0.3335 UAS: 93.47% LAS: 91.27%
0:04:48.040063s elapsed

Epoch 9 / 1000:
train: Loss: 0.2487 UAS: 94.81% LAS: 92.96%
dev:   Loss: 0.3215 UAS: 93.70% LAS: 91.49%
test:  Loss: 0.3268 UAS: 93.65% LAS: 91.54%
0:04:47.494597s elapsed

Epoch 10 / 1000:
train: Loss: 0.2374 UAS: 95.03% LAS: 93.24%
dev:   Loss: 0.3092 UAS: 93.90% LAS: 91.76%
test:  Loss: 0.3161 UAS: 93.87% LAS: 91.78%
0:04:48.581129s elapsed

Epoch 11 / 1000:
train: Loss: 0.2295 UAS: 95.14% LAS: 93.35%
dev:   Loss: 0.3051 UAS: 93.95% LAS: 91.84%
test:  Loss: 0.3068 UAS: 94.03% LAS: 91.96%
0:04:48.347951s elapsed

Epoch 12 / 1000:
train: Loss: 0.2210 UAS: 95.38% LAS: 93.62%
dev:   Loss: 0.3011 UAS: 94.03% LAS: 92.00%
test:  Loss: 0.3044 UAS: 94.14% LAS: 92.12%
0:04:47.523927s elapsed

Epoch 13 / 1000:
train: Loss: 0.2150 UAS: 95.49% LAS: 93.75%
dev:   Loss: 0.2998 UAS: 94.14% LAS: 92.11%
test:  Loss: 0.2990 UAS: 94.29% LAS: 92.28%
0:04:49.291105s elapsed

Epoch 14 / 1000:
train: Loss: 0.2115 UAS: 95.55% LAS: 93.83%
dev:   Loss: 0.2912 UAS: 94.11% LAS: 92.04%
test:  Loss: 0.2939 UAS: 94.33% LAS: 92.33%
0:04:48.410157s elapsed

Epoch 15 / 1000:
train: Loss: 0.2051 UAS: 95.71% LAS: 93.99%
dev:   Loss: 0.2877 UAS: 94.29% LAS: 92.15%
test:  Loss: 0.2942 UAS: 94.39% LAS: 92.41%
0:04:47.731747s elapsed

Epoch 16 / 1000:
train: Loss: 0.2000 UAS: 95.78% LAS: 94.08%
dev:   Loss: 0.2925 UAS: 94.24% LAS: 92.22%
test:  Loss: 0.2961 UAS: 94.46% LAS: 92.49%
0:04:49.171188s elapsed

Epoch 17 / 1000:
train: Loss: 0.1968 UAS: 95.88% LAS: 94.19%
dev:   Loss: 0.2846 UAS: 94.39% LAS: 92.33%
test:  Loss: 0.2903 UAS: 94.54% LAS: 92.55%
0:04:48.184200s elapsed

Epoch 18 / 1000:
train: Loss: 0.1918 UAS: 95.94% LAS: 94.27%
dev:   Loss: 0.2801 UAS: 94.42% LAS: 92.30%
test:  Loss: 0.2852 UAS: 94.52% LAS: 92.52%
0:04:46.728155s elapsed

Epoch 19 / 1000:
train: Loss: 0.1877 UAS: 96.07% LAS: 94.40%
dev:   Loss: 0.2795 UAS: 94.56% LAS: 92.46%
test:  Loss: 0.2831 UAS: 94.79% LAS: 92.83%
0:04:47.755601s elapsed

Epoch 20 / 1000:
train: Loss: 0.1855 UAS: 96.08% LAS: 94.44%
dev:   Loss: 0.2813 UAS: 94.51% LAS: 92.53%
test:  Loss: 0.2822 UAS: 94.64% LAS: 92.71%
0:04:47.586496s elapsed

Epoch 21 / 1000:
train: Loss: 0.1807 UAS: 96.25% LAS: 94.59%
dev:   Loss: 0.2760 UAS: 94.47% LAS: 92.45%
test:  Loss: 0.2795 UAS: 94.77% LAS: 92.84%
0:04:45.967916s elapsed

Epoch 22 / 1000:
train: Loss: 0.1777 UAS: 96.26% LAS: 94.64%
dev:   Loss: 0.2756 UAS: 94.58% LAS: 92.57%
test:  Loss: 0.2748 UAS: 94.84% LAS: 92.92%
0:04:46.402081s elapsed

Epoch 23 / 1000:
train: Loss: 0.1745 UAS: 96.34% LAS: 94.73%
dev:   Loss: 0.2747 UAS: 94.54% LAS: 92.54%
test:  Loss: 0.2780 UAS: 94.79% LAS: 92.95%
0:04:47.380539s elapsed

Epoch 24 / 1000:
train: Loss: 0.1708 UAS: 96.41% LAS: 94.82%
dev:   Loss: 0.2684 UAS: 94.61% LAS: 92.61%
test:  Loss: 0.2729 UAS: 94.89% LAS: 92.96%
0:04:46.774590s elapsed

Epoch 25 / 1000:
train: Loss: 0.1682 UAS: 96.44% LAS: 94.86%
dev:   Loss: 0.2687 UAS: 94.67% LAS: 92.69%
test:  Loss: 0.2767 UAS: 94.93% LAS: 93.01%
0:04:45.871716s elapsed

Epoch 26 / 1000:
train: Loss: 0.1652 UAS: 96.55% LAS: 94.97%
dev:   Loss: 0.2705 UAS: 94.80% LAS: 92.85%
test:  Loss: 0.2744 UAS: 94.93% LAS: 93.03%
0:04:46.219829s elapsed

Epoch 27 / 1000:
train: Loss: 0.1628 UAS: 96.60% LAS: 95.02%
dev:   Loss: 0.2659 UAS: 94.70% LAS: 92.76%
test:  Loss: 0.2685 UAS: 94.99% LAS: 93.10%
0:04:47.194986s elapsed

Epoch 28 / 1000:
train: Loss: 0.1625 UAS: 96.61% LAS: 95.05%
dev:   Loss: 0.2639 UAS: 94.72% LAS: 92.76%
test:  Loss: 0.2640 UAS: 95.04% LAS: 93.14%
0:04:46.246347s elapsed

Epoch 29 / 1000:
train: Loss: 0.1584 UAS: 96.69% LAS: 95.13%
dev:   Loss: 0.2666 UAS: 94.66% LAS: 92.67%
test:  Loss: 0.2667 UAS: 95.06% LAS: 93.19%
0:04:46.786486s elapsed

Epoch 30 / 1000:
train: Loss: 0.1552 UAS: 96.73% LAS: 95.17%
dev:   Loss: 0.2615 UAS: 94.83% LAS: 92.94%
test:  Loss: 0.2678 UAS: 95.11% LAS: 93.24%
0:04:47.569226s elapsed

Epoch 31 / 1000:
train: Loss: 0.1543 UAS: 96.76% LAS: 95.22%
dev:   Loss: 0.2616 UAS: 94.87% LAS: 92.90%
test:  Loss: 0.2625 UAS: 95.12% LAS: 93.26%
0:04:47.600821s elapsed

Epoch 32 / 1000:
train: Loss: 0.1512 UAS: 96.83% LAS: 95.28%
dev:   Loss: 0.2625 UAS: 94.82% LAS: 92.84%
test:  Loss: 0.2678 UAS: 95.09% LAS: 93.26%
0:04:46.480805s elapsed

Epoch 33 / 1000:
train: Loss: 0.1501 UAS: 96.85% LAS: 95.31%
dev:   Loss: 0.2611 UAS: 94.92% LAS: 93.02%
test:  Loss: 0.2612 UAS: 95.20% LAS: 93.36%
0:04:46.575422s elapsed

Epoch 34 / 1000:
train: Loss: 0.1482 UAS: 96.90% LAS: 95.37%
dev:   Loss: 0.2596 UAS: 94.81% LAS: 92.78%
test:  Loss: 0.2638 UAS: 95.18% LAS: 93.32%
0:04:47.185587s elapsed

Epoch 35 / 1000:
train: Loss: 0.1470 UAS: 96.93% LAS: 95.41%
dev:   Loss: 0.2566 UAS: 94.90% LAS: 93.01%
test:  Loss: 0.2582 UAS: 95.20% LAS: 93.38%
0:04:47.108433s elapsed

Epoch 36 / 1000:
train: Loss: 0.1448 UAS: 96.97% LAS: 95.47%
dev:   Loss: 0.2551 UAS: 94.92% LAS: 92.93%
test:  Loss: 0.2595 UAS: 95.25% LAS: 93.43%
0:04:46.880405s elapsed

Epoch 37 / 1000:
train: Loss: 0.1420 UAS: 97.04% LAS: 95.53%
dev:   Loss: 0.2547 UAS: 94.94% LAS: 93.02%
test:  Loss: 0.2594 UAS: 95.25% LAS: 93.45%
0:04:47.264771s elapsed

Epoch 38 / 1000:
train: Loss: 0.1407 UAS: 97.07% LAS: 95.58%
dev:   Loss: 0.2520 UAS: 95.08% LAS: 93.18%
test:  Loss: 0.2573 UAS: 95.24% LAS: 93.45%
0:04:45.562709s elapsed

Epoch 39 / 1000:
train: Loss: 0.1396 UAS: 97.08% LAS: 95.59%
dev:   Loss: 0.2528 UAS: 94.96% LAS: 93.07%
test:  Loss: 0.2602 UAS: 95.23% LAS: 93.45%
0:04:46.095510s elapsed

Epoch 40 / 1000:
train: Loss: 0.1367 UAS: 97.14% LAS: 95.67%
dev:   Loss: 0.2543 UAS: 94.98% LAS: 93.09%
test:  Loss: 0.2620 UAS: 95.28% LAS: 93.51%
0:04:46.623421s elapsed

Epoch 41 / 1000:
train: Loss: 0.1363 UAS: 97.15% LAS: 95.68%
dev:   Loss: 0.2563 UAS: 94.98% LAS: 93.07%
test:  Loss: 0.2639 UAS: 95.28% LAS: 93.48%
0:04:46.945636s elapsed

Epoch 42 / 1000:
train: Loss: 0.1358 UAS: 97.19% LAS: 95.71%
dev:   Loss: 0.2526 UAS: 95.04% LAS: 93.11%
test:  Loss: 0.2578 UAS: 95.39% LAS: 93.58%
0:04:46.456093s elapsed

Epoch 43 / 1000:
train: Loss: 0.1332 UAS: 97.24% LAS: 95.77%
dev:   Loss: 0.2574 UAS: 94.93% LAS: 92.96%
test:  Loss: 0.2556 UAS: 95.42% LAS: 93.59%
0:04:47.015227s elapsed

Epoch 44 / 1000:
train: Loss: 0.1325 UAS: 97.25% LAS: 95.78%
dev:   Loss: 0.2560 UAS: 95.04% LAS: 93.08%
test:  Loss: 0.2600 UAS: 95.39% LAS: 93.59%
0:04:46.114931s elapsed

Epoch 45 / 1000:
train: Loss: 0.1311 UAS: 97.28% LAS: 95.82%
dev:   Loss: 0.2497 UAS: 95.12% LAS: 93.17%
test:  Loss: 0.2514 UAS: 95.41% LAS: 93.62%
0:04:47.747439s elapsed

Epoch 46 / 1000:
train: Loss: 0.1291 UAS: 97.31% LAS: 95.86%
dev:   Loss: 0.2492 UAS: 95.11% LAS: 93.24%
test:  Loss: 0.2516 UAS: 95.46% LAS: 93.67%
0:04:48.144532s elapsed

Epoch 47 / 1000:
train: Loss: 0.1284 UAS: 97.34% LAS: 95.89%
dev:   Loss: 0.2531 UAS: 95.05% LAS: 93.15%
test:  Loss: 0.2523 UAS: 95.44% LAS: 93.66%
0:04:46.933864s elapsed

Epoch 48 / 1000:
train: Loss: 0.1269 UAS: 97.34% LAS: 95.91%
dev:   Loss: 0.2510 UAS: 95.01% LAS: 93.09%
test:  Loss: 0.2552 UAS: 95.46% LAS: 93.69%
0:04:48.555357s elapsed

Epoch 49 / 1000:
train: Loss: 0.1252 UAS: 97.39% LAS: 95.97%
dev:   Loss: 0.2500 UAS: 95.12% LAS: 93.25%
test:  Loss: 0.2541 UAS: 95.48% LAS: 93.70%
0:04:47.102155s elapsed

Epoch 50 / 1000:
train: Loss: 0.1251 UAS: 97.41% LAS: 95.98%
dev:   Loss: 0.2503 UAS: 95.03% LAS: 93.17%
test:  Loss: 0.2514 UAS: 95.39% LAS: 93.62%
0:04:46.450310s elapsed

Epoch 51 / 1000:
train: Loss: 0.1240 UAS: 97.44% LAS: 96.01%
dev:   Loss: 0.2479 UAS: 95.15% LAS: 93.27%
test:  Loss: 0.2515 UAS: 95.44% LAS: 93.68%
0:04:47.116271s elapsed

Epoch 52 / 1000:
train: Loss: 0.1229 UAS: 97.46% LAS: 96.05%
dev:   Loss: 0.2500 UAS: 95.20% LAS: 93.30%
test:  Loss: 0.2580 UAS: 95.43% LAS: 93.66%
0:04:48.352825s elapsed

Epoch 53 / 1000:
train: Loss: 0.1212 UAS: 97.49% LAS: 96.09%
dev:   Loss: 0.2466 UAS: 95.14% LAS: 93.29%
test:  Loss: 0.2533 UAS: 95.42% LAS: 93.68%
0:04:46.365640s elapsed

Epoch 54 / 1000:
train: Loss: 0.1205 UAS: 97.52% LAS: 96.12%
dev:   Loss: 0.2502 UAS: 95.23% LAS: 93.35%
test:  Loss: 0.2573 UAS: 95.44% LAS: 93.69%
0:04:48.016981s elapsed

Epoch 55 / 1000:
train: Loss: 0.1193 UAS: 97.56% LAS: 96.17%
dev:   Loss: 0.2448 UAS: 95.20% LAS: 93.36%
test:  Loss: 0.2508 UAS: 95.44% LAS: 93.69%
0:04:45.431583s elapsed

Epoch 56 / 1000:
train: Loss: 0.1175 UAS: 97.57% LAS: 96.18%
dev:   Loss: 0.2490 UAS: 95.17% LAS: 93.27%
test:  Loss: 0.2546 UAS: 95.49% LAS: 93.74%
0:04:47.385363s elapsed

Epoch 57 / 1000:
train: Loss: 0.1170 UAS: 97.59% LAS: 96.19%
dev:   Loss: 0.2467 UAS: 95.22% LAS: 93.33%
test:  Loss: 0.2530 UAS: 95.50% LAS: 93.77%
0:04:46.116784s elapsed

Epoch 58 / 1000:
train: Loss: 0.1157 UAS: 97.62% LAS: 96.23%
dev:   Loss: 0.2468 UAS: 95.26% LAS: 93.37%
test:  Loss: 0.2500 UAS: 95.54% LAS: 93.83%
0:04:46.800317s elapsed

Epoch 59 / 1000:
train: Loss: 0.1145 UAS: 97.66% LAS: 96.29%
dev:   Loss: 0.2453 UAS: 95.22% LAS: 93.36%
test:  Loss: 0.2498 UAS: 95.53% LAS: 93.80%
0:04:45.518934s elapsed

Epoch 60 / 1000:
train: Loss: 0.1130 UAS: 97.66% LAS: 96.28%
dev:   Loss: 0.2463 UAS: 95.18% LAS: 93.28%
test:  Loss: 0.2515 UAS: 95.56% LAS: 93.85%
0:04:46.484679s elapsed

Epoch 61 / 1000:
train: Loss: 0.1132 UAS: 97.69% LAS: 96.32%
dev:   Loss: 0.2450 UAS: 95.30% LAS: 93.39%
test:  Loss: 0.2503 UAS: 95.52% LAS: 93.80%
0:04:47.690503s elapsed

Epoch 62 / 1000:
train: Loss: 0.1123 UAS: 97.68% LAS: 96.32%
dev:   Loss: 0.2457 UAS: 95.18% LAS: 93.34%
test:  Loss: 0.2510 UAS: 95.51% LAS: 93.78%
0:04:47.559678s elapsed

Epoch 63 / 1000:
train: Loss: 0.1119 UAS: 97.71% LAS: 96.35%
dev:   Loss: 0.2459 UAS: 95.25% LAS: 93.33%
test:  Loss: 0.2461 UAS: 95.59% LAS: 93.84%
0:04:46.314970s elapsed

Epoch 64 / 1000:
train: Loss: 0.1109 UAS: 97.73% LAS: 96.37%
dev:   Loss: 0.2442 UAS: 95.22% LAS: 93.34%
test:  Loss: 0.2481 UAS: 95.61% LAS: 93.88%
0:04:46.283817s elapsed

Epoch 65 / 1000:
train: Loss: 0.1094 UAS: 97.74% LAS: 96.39%
dev:   Loss: 0.2471 UAS: 95.25% LAS: 93.40%
test:  Loss: 0.2527 UAS: 95.56% LAS: 93.82%
0:04:44.671502s elapsed

Epoch 66 / 1000:
train: Loss: 0.1088 UAS: 97.75% LAS: 96.41%
dev:   Loss: 0.2460 UAS: 95.28% LAS: 93.40%
test:  Loss: 0.2529 UAS: 95.55% LAS: 93.82%
0:04:47.550637s elapsed

Epoch 67 / 1000:
train: Loss: 0.1074 UAS: 97.79% LAS: 96.45%
dev:   Loss: 0.2456 UAS: 95.27% LAS: 93.45%
test:  Loss: 0.2486 UAS: 95.59% LAS: 93.89%
0:04:45.954847s elapsed

Epoch 68 / 1000:
train: Loss: 0.1070 UAS: 97.81% LAS: 96.48%
dev:   Loss: 0.2434 UAS: 95.29% LAS: 93.43%
test:  Loss: 0.2448 UAS: 95.59% LAS: 93.87%
0:04:46.070946s elapsed

Epoch 69 / 1000:
train: Loss: 0.1068 UAS: 97.82% LAS: 96.48%
dev:   Loss: 0.2439 UAS: 95.24% LAS: 93.38%
test:  Loss: 0.2465 UAS: 95.64% LAS: 93.90%
0:04:46.494809s elapsed

Epoch 70 / 1000:
train: Loss: 0.1056 UAS: 97.86% LAS: 96.52%
dev:   Loss: 0.2426 UAS: 95.30% LAS: 93.45%
test:  Loss: 0.2477 UAS: 95.63% LAS: 93.93%
0:04:46.900390s elapsed

Epoch 71 / 1000:
train: Loss: 0.1047 UAS: 97.86% LAS: 96.53%
dev:   Loss: 0.2429 UAS: 95.30% LAS: 93.44%
test:  Loss: 0.2464 UAS: 95.66% LAS: 93.98%
0:04:46.098104s elapsed

Epoch 72 / 1000:
train: Loss: 0.1041 UAS: 97.86% LAS: 96.54%
dev:   Loss: 0.2436 UAS: 95.31% LAS: 93.51%
test:  Loss: 0.2473 UAS: 95.66% LAS: 93.96%
0:04:45.366783s elapsed

Epoch 73 / 1000:
train: Loss: 0.1032 UAS: 97.90% LAS: 96.58%
dev:   Loss: 0.2447 UAS: 95.28% LAS: 93.44%
test:  Loss: 0.2488 UAS: 95.66% LAS: 93.93%
0:04:46.889630s elapsed

Epoch 74 / 1000:
train: Loss: 0.1022 UAS: 97.92% LAS: 96.60%
dev:   Loss: 0.2424 UAS: 95.25% LAS: 93.41%
test:  Loss: 0.2458 UAS: 95.69% LAS: 93.98%
0:04:46.968269s elapsed

Epoch 75 / 1000:
train: Loss: 0.1022 UAS: 97.93% LAS: 96.61%
dev:   Loss: 0.2418 UAS: 95.27% LAS: 93.45%
test:  Loss: 0.2469 UAS: 95.69% LAS: 93.99%
0:04:45.493237s elapsed

Epoch 76 / 1000:
train: Loss: 0.1012 UAS: 97.93% LAS: 96.61%
dev:   Loss: 0.2443 UAS: 95.36% LAS: 93.48%
test:  Loss: 0.2471 UAS: 95.67% LAS: 93.95%
0:04:46.422839s elapsed

Epoch 77 / 1000:
train: Loss: 0.1001 UAS: 97.95% LAS: 96.64%
dev:   Loss: 0.2448 UAS: 95.31% LAS: 93.51%
test:  Loss: 0.2456 UAS: 95.67% LAS: 93.97%
0:04:47.529555s elapsed

Epoch 78 / 1000:
train: Loss: 0.1000 UAS: 97.98% LAS: 96.68%
dev:   Loss: 0.2409 UAS: 95.38% LAS: 93.54%
test:  Loss: 0.2451 UAS: 95.74% LAS: 94.01%
0:04:46.896078s elapsed

Epoch 79 / 1000:
train: Loss: 0.0988 UAS: 97.98% LAS: 96.69%
dev:   Loss: 0.2458 UAS: 95.38% LAS: 93.54%
test:  Loss: 0.2463 UAS: 95.68% LAS: 93.95%
0:04:46.806873s elapsed

Epoch 80 / 1000:
train: Loss: 0.0993 UAS: 98.00% LAS: 96.70%
dev:   Loss: 0.2432 UAS: 95.27% LAS: 93.42%
test:  Loss: 0.2464 UAS: 95.68% LAS: 93.96%
0:04:46.477691s elapsed

Epoch 81 / 1000:
train: Loss: 0.0978 UAS: 98.02% LAS: 96.72%
dev:   Loss: 0.2483 UAS: 95.25% LAS: 93.42%
test:  Loss: 0.2496 UAS: 95.65% LAS: 93.95%
0:04:45.850135s elapsed

Epoch 82 / 1000:
train: Loss: 0.0972 UAS: 98.03% LAS: 96.74%
dev:   Loss: 0.2451 UAS: 95.31% LAS: 93.42%
test:  Loss: 0.2499 UAS: 95.64% LAS: 93.95%
0:04:46.870574s elapsed

Epoch 83 / 1000:
train: Loss: 0.0967 UAS: 98.05% LAS: 96.76%
dev:   Loss: 0.2460 UAS: 95.29% LAS: 93.44%
test:  Loss: 0.2524 UAS: 95.62% LAS: 93.94%
0:04:47.426362s elapsed

Epoch 84 / 1000:
train: Loss: 0.0961 UAS: 98.07% LAS: 96.79%
dev:   Loss: 0.2456 UAS: 95.38% LAS: 93.54%
test:  Loss: 0.2497 UAS: 95.59% LAS: 93.93%
0:04:45.641189s elapsed

Epoch 85 / 1000:
train: Loss: 0.0954 UAS: 98.09% LAS: 96.81%
dev:   Loss: 0.2430 UAS: 95.35% LAS: 93.51%
test:  Loss: 0.2484 UAS: 95.66% LAS: 93.97%
0:04:47.060988s elapsed

Epoch 86 / 1000:
train: Loss: 0.0951 UAS: 98.10% LAS: 96.81%
dev:   Loss: 0.2424 UAS: 95.39% LAS: 93.53%
test:  Loss: 0.2465 UAS: 95.67% LAS: 93.95%
0:04:46.502453s elapsed

Epoch 87 / 1000:
train: Loss: 0.0936 UAS: 98.12% LAS: 96.83%
dev:   Loss: 0.2431 UAS: 95.34% LAS: 93.53%
test:  Loss: 0.2457 UAS: 95.66% LAS: 93.97%
0:04:47.444507s elapsed

Epoch 88 / 1000:
train: Loss: 0.0939 UAS: 98.12% LAS: 96.86%
dev:   Loss: 0.2413 UAS: 95.32% LAS: 93.43%
test:  Loss: 0.2468 UAS: 95.64% LAS: 93.95%
0:04:46.470382s elapsed

Epoch 89 / 1000:
train: Loss: 0.0922 UAS: 98.15% LAS: 96.89%
dev:   Loss: 0.2430 UAS: 95.38% LAS: 93.50%
test:  Loss: 0.2465 UAS: 95.75% LAS: 94.08%
0:04:46.317000s elapsed

Epoch 90 / 1000:
train: Loss: 0.0922 UAS: 98.14% LAS: 96.89%
dev:   Loss: 0.2454 UAS: 95.32% LAS: 93.47%
test:  Loss: 0.2464 UAS: 95.68% LAS: 94.01%
0:04:45.361058s elapsed

Epoch 91 / 1000:
train: Loss: 0.0919 UAS: 98.17% LAS: 96.92%
dev:   Loss: 0.2412 UAS: 95.42% LAS: 93.55%
test:  Loss: 0.2479 UAS: 95.69% LAS: 93.98%
0:04:45.345072s elapsed

Epoch 92 / 1000:
train: Loss: 0.0916 UAS: 98.18% LAS: 96.93%
dev:   Loss: 0.2408 UAS: 95.39% LAS: 93.51%
test:  Loss: 0.2462 UAS: 95.76% LAS: 94.07%
0:04:47.148426s elapsed

Epoch 93 / 1000:
train: Loss: 0.0905 UAS: 98.19% LAS: 96.95%
dev:   Loss: 0.2413 UAS: 95.38% LAS: 93.53%
test:  Loss: 0.2485 UAS: 95.70% LAS: 94.00%
0:04:46.594893s elapsed

Epoch 94 / 1000:
train: Loss: 0.0911 UAS: 98.20% LAS: 96.95%
dev:   Loss: 0.2414 UAS: 95.39% LAS: 93.47%
test:  Loss: 0.2453 UAS: 95.65% LAS: 94.00%
0:04:46.890962s elapsed

Epoch 95 / 1000:
train: Loss: 0.0897 UAS: 98.21% LAS: 96.97%
dev:   Loss: 0.2451 UAS: 95.36% LAS: 93.50%
test:  Loss: 0.2491 UAS: 95.69% LAS: 94.03%
0:04:47.390714s elapsed

Epoch 96 / 1000:
train: Loss: 0.0890 UAS: 98.23% LAS: 97.00%
dev:   Loss: 0.2419 UAS: 95.37% LAS: 93.49%
test:  Loss: 0.2477 UAS: 95.71% LAS: 94.01%
0:04:46.480799s elapsed

Epoch 97 / 1000:
train: Loss: 0.0889 UAS: 98.22% LAS: 96.98%
dev:   Loss: 0.2412 UAS: 95.46% LAS: 93.59%
test:  Loss: 0.2466 UAS: 95.69% LAS: 94.01%
0:04:46.912906s elapsed

Epoch 98 / 1000:
train: Loss: 0.0893 UAS: 98.23% LAS: 97.00%
dev:   Loss: 0.2419 UAS: 95.38% LAS: 93.49%
test:  Loss: 0.2454 UAS: 95.70% LAS: 94.01%
0:04:46.115857s elapsed

Epoch 99 / 1000:
train: Loss: 0.0873 UAS: 98.26% LAS: 97.03%
dev:   Loss: 0.2465 UAS: 95.38% LAS: 93.53%
test:  Loss: 0.2509 UAS: 95.71% LAS: 94.02%
0:04:46.539069s elapsed

Epoch 100 / 1000:
train: Loss: 0.0882 UAS: 98.25% LAS: 97.02%
dev:   Loss: 0.2437 UAS: 95.38% LAS: 93.51%
test:  Loss: 0.2456 UAS: 95.71% LAS: 94.05%
0:04:46.326254s elapsed

Epoch 101 / 1000:
train: Loss: 0.0865 UAS: 98.28% LAS: 97.05%
dev:   Loss: 0.2460 UAS: 95.44% LAS: 93.60%
test:  Loss: 0.2513 UAS: 95.72% LAS: 94.01%
0:04:45.994487s elapsed

Epoch 102 / 1000:
train: Loss: 0.0866 UAS: 98.28% LAS: 97.06%
dev:   Loss: 0.2416 UAS: 95.40% LAS: 93.56%
test:  Loss: 0.2460 UAS: 95.68% LAS: 93.99%
0:04:43.707678s elapsed

Epoch 103 / 1000:
train: Loss: 0.0863 UAS: 98.30% LAS: 97.08%
dev:   Loss: 0.2425 UAS: 95.44% LAS: 93.57%
test:  Loss: 0.2476 UAS: 95.73% LAS: 94.01%
0:04:46.803726s elapsed

Epoch 104 / 1000:
train: Loss: 0.0860 UAS: 98.30% LAS: 97.08%
dev:   Loss: 0.2415 UAS: 95.48% LAS: 93.64%
test:  Loss: 0.2475 UAS: 95.69% LAS: 94.03%
0:04:49.834127s elapsed

Epoch 105 / 1000:
train: Loss: 0.0855 UAS: 98.32% LAS: 97.10%
dev:   Loss: 0.2424 UAS: 95.45% LAS: 93.59%
test:  Loss: 0.2500 UAS: 95.69% LAS: 93.98%
0:04:46.955824s elapsed

Epoch 106 / 1000:
train: Loss: 0.0851 UAS: 98.32% LAS: 97.11%
dev:   Loss: 0.2421 UAS: 95.43% LAS: 93.54%
test:  Loss: 0.2484 UAS: 95.68% LAS: 94.00%
0:04:46.568551s elapsed

Epoch 107 / 1000:
train: Loss: 0.0842 UAS: 98.33% LAS: 97.13%
dev:   Loss: 0.2431 UAS: 95.46% LAS: 93.64%
test:  Loss: 0.2503 UAS: 95.71% LAS: 94.01%
0:04:45.827562s elapsed

Epoch 108 / 1000:
train: Loss: 0.0843 UAS: 98.34% LAS: 97.13%
dev:   Loss: 0.2415 UAS: 95.47% LAS: 93.65%
test:  Loss: 0.2460 UAS: 95.69% LAS: 93.99%
0:04:46.538478s elapsed

Epoch 109 / 1000:
train: Loss: 0.0835 UAS: 98.36% LAS: 97.15%
dev:   Loss: 0.2438 UAS: 95.38% LAS: 93.56%
test:  Loss: 0.2485 UAS: 95.77% LAS: 94.06%
0:04:44.980415s elapsed

Epoch 110 / 1000:
train: Loss: 0.0833 UAS: 98.37% LAS: 97.16%
dev:   Loss: 0.2443 UAS: 95.43% LAS: 93.62%
test:  Loss: 0.2489 UAS: 95.74% LAS: 94.05%
0:04:47.224559s elapsed

Epoch 111 / 1000:
train: Loss: 0.0832 UAS: 98.37% LAS: 97.16%
dev:   Loss: 0.2432 UAS: 95.42% LAS: 93.57%
test:  Loss: 0.2456 UAS: 95.78% LAS: 94.11%
0:04:46.001341s elapsed

Epoch 112 / 1000:
train: Loss: 0.0826 UAS: 98.39% LAS: 97.19%
dev:   Loss: 0.2445 UAS: 95.45% LAS: 93.60%
test:  Loss: 0.2492 UAS: 95.74% LAS: 94.03%
0:04:45.488661s elapsed

Epoch 113 / 1000:
train: Loss: 0.0832 UAS: 98.40% LAS: 97.21%
dev:   Loss: 0.2412 UAS: 95.40% LAS: 93.58%
test:  Loss: 0.2426 UAS: 95.74% LAS: 94.06%
0:04:46.019171s elapsed

Epoch 114 / 1000:
train: Loss: 0.0814 UAS: 98.42% LAS: 97.22%
dev:   Loss: 0.2434 UAS: 95.47% LAS: 93.59%
test:  Loss: 0.2476 UAS: 95.74% LAS: 94.04%
0:04:46.487058s elapsed

Epoch 115 / 1000:
train: Loss: 0.0817 UAS: 98.42% LAS: 97.23%
dev:   Loss: 0.2419 UAS: 95.47% LAS: 93.62%
test:  Loss: 0.2450 UAS: 95.78% LAS: 94.10%
0:04:46.892598s elapsed

Epoch 116 / 1000:
train: Loss: 0.0815 UAS: 98.43% LAS: 97.24%
dev:   Loss: 0.2397 UAS: 95.52% LAS: 93.66%
test:  Loss: 0.2439 UAS: 95.80% LAS: 94.09%
0:04:46.819066s elapsed

Epoch 117 / 1000:
train: Loss: 0.0806 UAS: 98.44% LAS: 97.25%
dev:   Loss: 0.2446 UAS: 95.43% LAS: 93.59%
test:  Loss: 0.2494 UAS: 95.78% LAS: 94.09%
0:04:45.885259s elapsed

Epoch 118 / 1000:
train: Loss: 0.0805 UAS: 98.45% LAS: 97.27%
dev:   Loss: 0.2419 UAS: 95.50% LAS: 93.61%
test:  Loss: 0.2459 UAS: 95.78% LAS: 94.08%
0:04:45.362189s elapsed

Epoch 119 / 1000:
train: Loss: 0.0801 UAS: 98.44% LAS: 97.27%
dev:   Loss: 0.2429 UAS: 95.44% LAS: 93.59%
test:  Loss: 0.2481 UAS: 95.77% LAS: 94.09%
0:04:46.631143s elapsed

Epoch 120 / 1000:
train: Loss: 0.0798 UAS: 98.46% LAS: 97.28%
dev:   Loss: 0.2423 UAS: 95.41% LAS: 93.54%
test:  Loss: 0.2503 UAS: 95.77% LAS: 94.06%
0:04:46.977957s elapsed

Epoch 121 / 1000:
train: Loss: 0.0787 UAS: 98.48% LAS: 97.30%
dev:   Loss: 0.2435 UAS: 95.46% LAS: 93.61%
test:  Loss: 0.2511 UAS: 95.77% LAS: 94.10%
0:04:46.404128s elapsed

Epoch 122 / 1000:
train: Loss: 0.0789 UAS: 98.49% LAS: 97.32%
dev:   Loss: 0.2430 UAS: 95.43% LAS: 93.57%
test:  Loss: 0.2475 UAS: 95.77% LAS: 94.07%
0:04:46.715978s elapsed

Epoch 123 / 1000:
train: Loss: 0.0779 UAS: 98.50% LAS: 97.32%
dev:   Loss: 0.2458 UAS: 95.47% LAS: 93.63%
test:  Loss: 0.2510 UAS: 95.75% LAS: 94.04%
0:04:47.455120s elapsed

Epoch 124 / 1000:
train: Loss: 0.0781 UAS: 98.49% LAS: 97.32%
dev:   Loss: 0.2439 UAS: 95.48% LAS: 93.64%
test:  Loss: 0.2496 UAS: 95.78% LAS: 94.08%
0:04:46.316383s elapsed

Epoch 125 / 1000:
train: Loss: 0.0781 UAS: 98.50% LAS: 97.33%
dev:   Loss: 0.2443 UAS: 95.45% LAS: 93.62%
test:  Loss: 0.2473 UAS: 95.81% LAS: 94.11%
0:04:45.260325s elapsed

Epoch 126 / 1000:
train: Loss: 0.0774 UAS: 98.50% LAS: 97.34%
dev:   Loss: 0.2439 UAS: 95.51% LAS: 93.67%
test:  Loss: 0.2466 UAS: 95.77% LAS: 94.11%
0:04:45.736747s elapsed

Epoch 127 / 1000:
train: Loss: 0.0772 UAS: 98.52% LAS: 97.36%
dev:   Loss: 0.2401 UAS: 95.56% LAS: 93.71%
test:  Loss: 0.2495 UAS: 95.81% LAS: 94.12%
0:04:46.747881s elapsed

Epoch 128 / 1000:
train: Loss: 0.0770 UAS: 98.52% LAS: 97.36%
dev:   Loss: 0.2438 UAS: 95.46% LAS: 93.58%
test:  Loss: 0.2495 UAS: 95.81% LAS: 94.12%
0:04:46.452620s elapsed

Epoch 129 / 1000:
train: Loss: 0.0768 UAS: 98.53% LAS: 97.37%
dev:   Loss: 0.2443 UAS: 95.49% LAS: 93.62%
test:  Loss: 0.2497 UAS: 95.77% LAS: 94.06%
0:04:47.221433s elapsed

Epoch 130 / 1000:
train: Loss: 0.0758 UAS: 98.54% LAS: 97.39%
dev:   Loss: 0.2452 UAS: 95.50% LAS: 93.66%
test:  Loss: 0.2491 UAS: 95.84% LAS: 94.14%
0:04:46.861664s elapsed

Epoch 131 / 1000:
train: Loss: 0.0763 UAS: 98.54% LAS: 97.39%
dev:   Loss: 0.2444 UAS: 95.47% LAS: 93.63%
test:  Loss: 0.2471 UAS: 95.78% LAS: 94.10%
0:04:47.141123s elapsed

Epoch 132 / 1000:
train: Loss: 0.0750 UAS: 98.54% LAS: 97.40%
dev:   Loss: 0.2468 UAS: 95.50% LAS: 93.61%
test:  Loss: 0.2496 UAS: 95.79% LAS: 94.10%
0:04:47.098372s elapsed

Epoch 133 / 1000:
train: Loss: 0.0750 UAS: 98.56% LAS: 97.42%
dev:   Loss: 0.2448 UAS: 95.44% LAS: 93.62%
test:  Loss: 0.2489 UAS: 95.80% LAS: 94.10%
0:04:46.500451s elapsed

Epoch 134 / 1000:
train: Loss: 0.0749 UAS: 98.57% LAS: 97.42%
dev:   Loss: 0.2425 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2476 UAS: 95.81% LAS: 94.14%
0:04:47.695537s elapsed

Epoch 135 / 1000:
train: Loss: 0.0744 UAS: 98.57% LAS: 97.44%
dev:   Loss: 0.2436 UAS: 95.46% LAS: 93.62%
test:  Loss: 0.2492 UAS: 95.81% LAS: 94.12%
0:04:45.643159s elapsed

Epoch 136 / 1000:
train: Loss: 0.0744 UAS: 98.57% LAS: 97.43%
dev:   Loss: 0.2443 UAS: 95.53% LAS: 93.70%
test:  Loss: 0.2489 UAS: 95.85% LAS: 94.17%
0:04:45.720389s elapsed

Epoch 137 / 1000:
train: Loss: 0.0744 UAS: 98.58% LAS: 97.45%
dev:   Loss: 0.2426 UAS: 95.54% LAS: 93.72%
test:  Loss: 0.2470 UAS: 95.79% LAS: 94.10%
0:04:44.866065s elapsed

Epoch 138 / 1000:
train: Loss: 0.0741 UAS: 98.60% LAS: 97.46%
dev:   Loss: 0.2416 UAS: 95.52% LAS: 93.69%
test:  Loss: 0.2467 UAS: 95.80% LAS: 94.15%
0:04:45.748381s elapsed

Epoch 139 / 1000:
train: Loss: 0.0736 UAS: 98.61% LAS: 97.48%
dev:   Loss: 0.2446 UAS: 95.51% LAS: 93.68%
test:  Loss: 0.2478 UAS: 95.79% LAS: 94.13%
0:04:46.537914s elapsed

Epoch 140 / 1000:
train: Loss: 0.0732 UAS: 98.61% LAS: 97.48%
dev:   Loss: 0.2436 UAS: 95.53% LAS: 93.72%
test:  Loss: 0.2482 UAS: 95.82% LAS: 94.16%
0:04:45.453875s elapsed

Epoch 141 / 1000:
train: Loss: 0.0730 UAS: 98.61% LAS: 97.49%
dev:   Loss: 0.2422 UAS: 95.51% LAS: 93.71%
test:  Loss: 0.2484 UAS: 95.77% LAS: 94.07%
0:04:47.523622s elapsed

Epoch 142 / 1000:
train: Loss: 0.0731 UAS: 98.61% LAS: 97.49%
dev:   Loss: 0.2423 UAS: 95.51% LAS: 93.65%
test:  Loss: 0.2503 UAS: 95.81% LAS: 94.13%
0:04:47.113939s elapsed

Epoch 143 / 1000:
train: Loss: 0.0726 UAS: 98.61% LAS: 97.49%
dev:   Loss: 0.2438 UAS: 95.53% LAS: 93.69%
test:  Loss: 0.2491 UAS: 95.77% LAS: 94.10%
0:04:45.707966s elapsed

Epoch 144 / 1000:
train: Loss: 0.0725 UAS: 98.63% LAS: 97.51%
dev:   Loss: 0.2425 UAS: 95.53% LAS: 93.66%
test:  Loss: 0.2486 UAS: 95.81% LAS: 94.15%
0:04:46.917898s elapsed

Epoch 145 / 1000:
train: Loss: 0.0717 UAS: 98.63% LAS: 97.52%
dev:   Loss: 0.2442 UAS: 95.56% LAS: 93.72%
test:  Loss: 0.2506 UAS: 95.82% LAS: 94.13%
0:04:46.523579s elapsed

Epoch 146 / 1000:
train: Loss: 0.0722 UAS: 98.64% LAS: 97.52%
dev:   Loss: 0.2417 UAS: 95.53% LAS: 93.71%
test:  Loss: 0.2488 UAS: 95.79% LAS: 94.11%
0:04:46.567359s elapsed

Epoch 147 / 1000:
train: Loss: 0.0718 UAS: 98.63% LAS: 97.52%
dev:   Loss: 0.2453 UAS: 95.49% LAS: 93.66%
test:  Loss: 0.2487 UAS: 95.78% LAS: 94.10%
0:04:46.473049s elapsed

Epoch 148 / 1000:
train: Loss: 0.0716 UAS: 98.64% LAS: 97.52%
dev:   Loss: 0.2448 UAS: 95.50% LAS: 93.65%
test:  Loss: 0.2508 UAS: 95.79% LAS: 94.11%
0:04:46.119973s elapsed

Epoch 149 / 1000:
train: Loss: 0.0713 UAS: 98.65% LAS: 97.54%
dev:   Loss: 0.2432 UAS: 95.53% LAS: 93.68%
test:  Loss: 0.2501 UAS: 95.79% LAS: 94.13%
0:04:47.146611s elapsed

Epoch 150 / 1000:
train: Loss: 0.0714 UAS: 98.66% LAS: 97.55%
dev:   Loss: 0.2436 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2474 UAS: 95.80% LAS: 94.15%
0:04:46.554642s elapsed

Epoch 151 / 1000:
train: Loss: 0.0706 UAS: 98.67% LAS: 97.56%
dev:   Loss: 0.2435 UAS: 95.53% LAS: 93.69%
test:  Loss: 0.2509 UAS: 95.78% LAS: 94.14%
0:04:44.989538s elapsed

Epoch 152 / 1000:
train: Loss: 0.0704 UAS: 98.68% LAS: 97.57%
dev:   Loss: 0.2452 UAS: 95.53% LAS: 93.68%
test:  Loss: 0.2508 UAS: 95.82% LAS: 94.15%
0:04:45.725189s elapsed

Epoch 153 / 1000:
train: Loss: 0.0700 UAS: 98.68% LAS: 97.57%
dev:   Loss: 0.2454 UAS: 95.49% LAS: 93.65%
test:  Loss: 0.2526 UAS: 95.75% LAS: 94.09%
0:04:46.171476s elapsed

Epoch 154 / 1000:
train: Loss: 0.0700 UAS: 98.68% LAS: 97.58%
dev:   Loss: 0.2445 UAS: 95.56% LAS: 93.71%
test:  Loss: 0.2500 UAS: 95.77% LAS: 94.10%
0:04:46.762862s elapsed

Epoch 155 / 1000:
train: Loss: 0.0697 UAS: 98.69% LAS: 97.59%
dev:   Loss: 0.2431 UAS: 95.53% LAS: 93.69%
test:  Loss: 0.2498 UAS: 95.76% LAS: 94.10%
0:04:44.811368s elapsed

Epoch 156 / 1000:
train: Loss: 0.0701 UAS: 98.69% LAS: 97.59%
dev:   Loss: 0.2443 UAS: 95.58% LAS: 93.71%
test:  Loss: 0.2501 UAS: 95.75% LAS: 94.09%
0:04:46.799342s elapsed

Epoch 157 / 1000:
train: Loss: 0.0692 UAS: 98.69% LAS: 97.60%
dev:   Loss: 0.2459 UAS: 95.57% LAS: 93.74%
test:  Loss: 0.2506 UAS: 95.75% LAS: 94.09%
0:04:46.097291s elapsed

Epoch 158 / 1000:
train: Loss: 0.0693 UAS: 98.71% LAS: 97.61%
dev:   Loss: 0.2440 UAS: 95.52% LAS: 93.68%
test:  Loss: 0.2496 UAS: 95.80% LAS: 94.14%
0:04:46.339334s elapsed

Epoch 159 / 1000:
train: Loss: 0.0686 UAS: 98.71% LAS: 97.63%
dev:   Loss: 0.2452 UAS: 95.51% LAS: 93.67%
test:  Loss: 0.2504 UAS: 95.77% LAS: 94.11%
0:04:47.333038s elapsed

Epoch 160 / 1000:
train: Loss: 0.0691 UAS: 98.71% LAS: 97.61%
dev:   Loss: 0.2444 UAS: 95.50% LAS: 93.66%
test:  Loss: 0.2488 UAS: 95.83% LAS: 94.17%
0:04:47.170088s elapsed

Epoch 161 / 1000:
train: Loss: 0.0688 UAS: 98.72% LAS: 97.62%
dev:   Loss: 0.2443 UAS: 95.53% LAS: 93.69%
test:  Loss: 0.2494 UAS: 95.79% LAS: 94.12%
0:04:46.234013s elapsed

Epoch 162 / 1000:
train: Loss: 0.0684 UAS: 98.72% LAS: 97.63%
dev:   Loss: 0.2468 UAS: 95.51% LAS: 93.69%
test:  Loss: 0.2505 UAS: 95.78% LAS: 94.12%
0:04:46.400395s elapsed

Epoch 163 / 1000:
train: Loss: 0.0684 UAS: 98.73% LAS: 97.63%
dev:   Loss: 0.2435 UAS: 95.50% LAS: 93.68%
test:  Loss: 0.2492 UAS: 95.78% LAS: 94.11%
0:04:46.771834s elapsed

Epoch 164 / 1000:
train: Loss: 0.0683 UAS: 98.73% LAS: 97.64%
dev:   Loss: 0.2434 UAS: 95.49% LAS: 93.66%
test:  Loss: 0.2473 UAS: 95.78% LAS: 94.11%
0:04:47.335519s elapsed

Epoch 165 / 1000:
train: Loss: 0.0681 UAS: 98.73% LAS: 97.64%
dev:   Loss: 0.2447 UAS: 95.52% LAS: 93.70%
test:  Loss: 0.2509 UAS: 95.80% LAS: 94.12%
0:04:46.495038s elapsed

Epoch 166 / 1000:
train: Loss: 0.0680 UAS: 98.73% LAS: 97.64%
dev:   Loss: 0.2451 UAS: 95.54% LAS: 93.71%
test:  Loss: 0.2508 UAS: 95.83% LAS: 94.17%
0:04:45.969541s elapsed

Epoch 167 / 1000:
train: Loss: 0.0676 UAS: 98.74% LAS: 97.66%
dev:   Loss: 0.2446 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2498 UAS: 95.80% LAS: 94.12%
0:04:45.832940s elapsed

Epoch 168 / 1000:
train: Loss: 0.0668 UAS: 98.75% LAS: 97.67%
dev:   Loss: 0.2463 UAS: 95.53% LAS: 93.69%
test:  Loss: 0.2527 UAS: 95.77% LAS: 94.08%
0:04:45.859354s elapsed

Epoch 169 / 1000:
train: Loss: 0.0672 UAS: 98.76% LAS: 97.68%
dev:   Loss: 0.2454 UAS: 95.50% LAS: 93.67%
test:  Loss: 0.2499 UAS: 95.82% LAS: 94.15%
0:04:46.435495s elapsed

Epoch 170 / 1000:
train: Loss: 0.0672 UAS: 98.76% LAS: 97.68%
dev:   Loss: 0.2443 UAS: 95.55% LAS: 93.72%
test:  Loss: 0.2498 UAS: 95.78% LAS: 94.11%
0:04:46.282566s elapsed

Epoch 171 / 1000:
train: Loss: 0.0666 UAS: 98.76% LAS: 97.69%
dev:   Loss: 0.2464 UAS: 95.51% LAS: 93.68%
test:  Loss: 0.2506 UAS: 95.79% LAS: 94.10%
0:04:47.182991s elapsed

Epoch 172 / 1000:
train: Loss: 0.0667 UAS: 98.76% LAS: 97.68%
dev:   Loss: 0.2467 UAS: 95.52% LAS: 93.68%
test:  Loss: 0.2503 UAS: 95.84% LAS: 94.16%
0:04:46.125292s elapsed

Epoch 173 / 1000:
train: Loss: 0.0664 UAS: 98.76% LAS: 97.69%
dev:   Loss: 0.2456 UAS: 95.50% LAS: 93.64%
test:  Loss: 0.2503 UAS: 95.84% LAS: 94.16%
0:04:46.031631s elapsed

Epoch 174 / 1000:
train: Loss: 0.0666 UAS: 98.76% LAS: 97.69%
dev:   Loss: 0.2455 UAS: 95.56% LAS: 93.71%
test:  Loss: 0.2504 UAS: 95.81% LAS: 94.13%
0:04:46.722376s elapsed

Epoch 175 / 1000:
train: Loss: 0.0661 UAS: 98.77% LAS: 97.70%
dev:   Loss: 0.2439 UAS: 95.52% LAS: 93.67%
test:  Loss: 0.2483 UAS: 95.81% LAS: 94.12%
0:04:45.728664s elapsed

Epoch 176 / 1000:
train: Loss: 0.0658 UAS: 98.78% LAS: 97.71%
dev:   Loss: 0.2452 UAS: 95.50% LAS: 93.67%
test:  Loss: 0.2515 UAS: 95.80% LAS: 94.13%
0:04:45.495771s elapsed

Epoch 177 / 1000:
train: Loss: 0.0657 UAS: 98.78% LAS: 97.71%
dev:   Loss: 0.2452 UAS: 95.55% LAS: 93.71%
test:  Loss: 0.2508 UAS: 95.80% LAS: 94.13%
0:04:46.469102s elapsed

Epoch 178 / 1000:
train: Loss: 0.0659 UAS: 98.79% LAS: 97.72%
dev:   Loss: 0.2444 UAS: 95.55% LAS: 93.71%
test:  Loss: 0.2483 UAS: 95.82% LAS: 94.14%
0:04:46.134651s elapsed

Epoch 179 / 1000:
train: Loss: 0.0653 UAS: 98.79% LAS: 97.72%
dev:   Loss: 0.2453 UAS: 95.54% LAS: 93.71%
test:  Loss: 0.2514 UAS: 95.79% LAS: 94.12%
0:04:45.087789s elapsed

Epoch 180 / 1000:
train: Loss: 0.0656 UAS: 98.79% LAS: 97.72%
dev:   Loss: 0.2431 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2483 UAS: 95.81% LAS: 94.14%
0:04:45.401094s elapsed

Epoch 181 / 1000:
train: Loss: 0.0653 UAS: 98.80% LAS: 97.73%
dev:   Loss: 0.2440 UAS: 95.53% LAS: 93.71%
test:  Loss: 0.2496 UAS: 95.83% LAS: 94.19%
0:04:46.073618s elapsed

Epoch 182 / 1000:
train: Loss: 0.0649 UAS: 98.80% LAS: 97.74%
dev:   Loss: 0.2442 UAS: 95.48% LAS: 93.66%
test:  Loss: 0.2504 UAS: 95.85% LAS: 94.18%
0:04:45.816693s elapsed

Epoch 183 / 1000:
train: Loss: 0.0649 UAS: 98.81% LAS: 97.76%
dev:   Loss: 0.2442 UAS: 95.50% LAS: 93.69%
test:  Loss: 0.2507 UAS: 95.84% LAS: 94.17%
0:04:46.791264s elapsed

Epoch 184 / 1000:
train: Loss: 0.0645 UAS: 98.81% LAS: 97.75%
dev:   Loss: 0.2458 UAS: 95.48% LAS: 93.66%
test:  Loss: 0.2512 UAS: 95.84% LAS: 94.15%
0:04:45.880033s elapsed

Epoch 185 / 1000:
train: Loss: 0.0640 UAS: 98.81% LAS: 97.76%
dev:   Loss: 0.2466 UAS: 95.51% LAS: 93.66%
test:  Loss: 0.2517 UAS: 95.82% LAS: 94.15%
0:04:47.103112s elapsed

Epoch 186 / 1000:
train: Loss: 0.0643 UAS: 98.81% LAS: 97.75%
dev:   Loss: 0.2465 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2522 UAS: 95.81% LAS: 94.15%
0:04:45.929677s elapsed

Epoch 187 / 1000:
train: Loss: 0.0642 UAS: 98.81% LAS: 97.76%
dev:   Loss: 0.2451 UAS: 95.57% LAS: 93.72%
test:  Loss: 0.2513 UAS: 95.86% LAS: 94.18%
0:04:45.766616s elapsed

Epoch 188 / 1000:
train: Loss: 0.0639 UAS: 98.82% LAS: 97.77%
dev:   Loss: 0.2457 UAS: 95.52% LAS: 93.70%
test:  Loss: 0.2506 UAS: 95.81% LAS: 94.15%
0:04:47.186231s elapsed

Epoch 189 / 1000:
train: Loss: 0.0636 UAS: 98.82% LAS: 97.78%
dev:   Loss: 0.2464 UAS: 95.52% LAS: 93.68%
test:  Loss: 0.2514 UAS: 95.85% LAS: 94.19%
0:04:45.795867s elapsed

Epoch 190 / 1000:
train: Loss: 0.0637 UAS: 98.83% LAS: 97.79%
dev:   Loss: 0.2467 UAS: 95.51% LAS: 93.66%
test:  Loss: 0.2507 UAS: 95.84% LAS: 94.17%
0:04:46.474314s elapsed

Epoch 191 / 1000:
train: Loss: 0.0636 UAS: 98.83% LAS: 97.79%
dev:   Loss: 0.2467 UAS: 95.49% LAS: 93.65%
test:  Loss: 0.2506 UAS: 95.84% LAS: 94.17%
0:04:47.117354s elapsed

Epoch 192 / 1000:
train: Loss: 0.0636 UAS: 98.83% LAS: 97.78%
dev:   Loss: 0.2467 UAS: 95.52% LAS: 93.70%
test:  Loss: 0.2516 UAS: 95.85% LAS: 94.18%
0:04:46.715243s elapsed

Epoch 193 / 1000:
train: Loss: 0.0632 UAS: 98.84% LAS: 97.79%
dev:   Loss: 0.2469 UAS: 95.50% LAS: 93.67%
test:  Loss: 0.2497 UAS: 95.85% LAS: 94.19%
0:04:46.385551s elapsed

Epoch 194 / 1000:
train: Loss: 0.0632 UAS: 98.84% LAS: 97.80%
dev:   Loss: 0.2467 UAS: 95.52% LAS: 93.67%
test:  Loss: 0.2499 UAS: 95.83% LAS: 94.15%
0:04:45.161527s elapsed

Epoch 195 / 1000:
train: Loss: 0.0631 UAS: 98.85% LAS: 97.81%
dev:   Loss: 0.2461 UAS: 95.51% LAS: 93.68%
test:  Loss: 0.2500 UAS: 95.85% LAS: 94.19%
0:04:46.684531s elapsed

Epoch 196 / 1000:
train: Loss: 0.0632 UAS: 98.84% LAS: 97.80%
dev:   Loss: 0.2456 UAS: 95.52% LAS: 93.67%
test:  Loss: 0.2497 UAS: 95.87% LAS: 94.19%
0:04:45.895134s elapsed

Epoch 197 / 1000:
train: Loss: 0.0629 UAS: 98.85% LAS: 97.81%
dev:   Loss: 0.2456 UAS: 95.58% LAS: 93.73%
test:  Loss: 0.2508 UAS: 95.86% LAS: 94.20%
0:04:46.783844s elapsed

Epoch 198 / 1000:
train: Loss: 0.0628 UAS: 98.86% LAS: 97.82%
dev:   Loss: 0.2462 UAS: 95.57% LAS: 93.74%
test:  Loss: 0.2496 UAS: 95.84% LAS: 94.19%
0:04:46.261181s elapsed

Epoch 199 / 1000:
train: Loss: 0.0625 UAS: 98.85% LAS: 97.81%
dev:   Loss: 0.2470 UAS: 95.55% LAS: 93.72%
test:  Loss: 0.2504 UAS: 95.85% LAS: 94.17%
0:04:44.610617s elapsed

Epoch 200 / 1000:
train: Loss: 0.0623 UAS: 98.86% LAS: 97.82%
dev:   Loss: 0.2471 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2509 UAS: 95.83% LAS: 94.16%
0:04:45.775271s elapsed

Epoch 201 / 1000:
train: Loss: 0.0624 UAS: 98.86% LAS: 97.82%
dev:   Loss: 0.2460 UAS: 95.58% LAS: 93.74%
test:  Loss: 0.2503 UAS: 95.83% LAS: 94.16%
0:04:46.280201s elapsed

Epoch 202 / 1000:
train: Loss: 0.0622 UAS: 98.86% LAS: 97.83%
dev:   Loss: 0.2464 UAS: 95.54% LAS: 93.68%
test:  Loss: 0.2506 UAS: 95.82% LAS: 94.15%
0:04:45.841011s elapsed

Epoch 203 / 1000:
train: Loss: 0.0622 UAS: 98.87% LAS: 97.83%
dev:   Loss: 0.2457 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2500 UAS: 95.84% LAS: 94.17%
0:04:46.841184s elapsed

Epoch 204 / 1000:
train: Loss: 0.0620 UAS: 98.87% LAS: 97.83%
dev:   Loss: 0.2472 UAS: 95.55% LAS: 93.68%
test:  Loss: 0.2505 UAS: 95.84% LAS: 94.18%
0:04:45.917466s elapsed

Epoch 205 / 1000:
train: Loss: 0.0617 UAS: 98.87% LAS: 97.84%
dev:   Loss: 0.2478 UAS: 95.53% LAS: 93.66%
test:  Loss: 0.2516 UAS: 95.82% LAS: 94.14%
0:04:46.600323s elapsed

Epoch 206 / 1000:
train: Loss: 0.0617 UAS: 98.87% LAS: 97.84%
dev:   Loss: 0.2469 UAS: 95.54% LAS: 93.69%
test:  Loss: 0.2514 UAS: 95.84% LAS: 94.15%
0:04:46.826506s elapsed

Epoch 207 / 1000:
train: Loss: 0.0619 UAS: 98.88% LAS: 97.85%
dev:   Loss: 0.2442 UAS: 95.60% LAS: 93.77%
test:  Loss: 0.2497 UAS: 95.84% LAS: 94.16%
0:04:46.405136s elapsed

Epoch 208 / 1000:
train: Loss: 0.0614 UAS: 98.88% LAS: 97.85%
dev:   Loss: 0.2462 UAS: 95.59% LAS: 93.75%
test:  Loss: 0.2521 UAS: 95.82% LAS: 94.11%
0:04:46.411581s elapsed

Epoch 209 / 1000:
train: Loss: 0.0613 UAS: 98.88% LAS: 97.86%
dev:   Loss: 0.2470 UAS: 95.57% LAS: 93.74%
test:  Loss: 0.2529 UAS: 95.86% LAS: 94.17%
0:04:46.638070s elapsed

Epoch 210 / 1000:
train: Loss: 0.0612 UAS: 98.89% LAS: 97.87%
dev:   Loss: 0.2464 UAS: 95.58% LAS: 93.76%
test:  Loss: 0.2543 UAS: 95.83% LAS: 94.17%
0:04:47.394790s elapsed

Epoch 211 / 1000:
train: Loss: 0.0613 UAS: 98.89% LAS: 97.86%
dev:   Loss: 0.2450 UAS: 95.56% LAS: 93.74%
test:  Loss: 0.2527 UAS: 95.84% LAS: 94.18%
0:04:45.524906s elapsed

Epoch 212 / 1000:
train: Loss: 0.0612 UAS: 98.89% LAS: 97.86%
dev:   Loss: 0.2455 UAS: 95.57% LAS: 93.74%
test:  Loss: 0.2531 UAS: 95.82% LAS: 94.16%
0:04:46.031227s elapsed

Epoch 213 / 1000:
train: Loss: 0.0612 UAS: 98.89% LAS: 97.86%
dev:   Loss: 0.2462 UAS: 95.54% LAS: 93.72%
test:  Loss: 0.2524 UAS: 95.82% LAS: 94.15%
0:04:44.411821s elapsed

Epoch 214 / 1000:
train: Loss: 0.0611 UAS: 98.90% LAS: 97.87%
dev:   Loss: 0.2453 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2516 UAS: 95.86% LAS: 94.19%
0:04:46.130134s elapsed

Epoch 215 / 1000:
train: Loss: 0.0608 UAS: 98.90% LAS: 97.88%
dev:   Loss: 0.2459 UAS: 95.54% LAS: 93.74%
test:  Loss: 0.2530 UAS: 95.85% LAS: 94.18%
0:04:45.426512s elapsed

Epoch 216 / 1000:
train: Loss: 0.0608 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2467 UAS: 95.51% LAS: 93.70%
test:  Loss: 0.2525 UAS: 95.87% LAS: 94.18%
0:04:45.573216s elapsed

Epoch 217 / 1000:
train: Loss: 0.0608 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2455 UAS: 95.55% LAS: 93.73%
test:  Loss: 0.2523 UAS: 95.81% LAS: 94.13%
0:04:45.619872s elapsed

Epoch 218 / 1000:
train: Loss: 0.0604 UAS: 98.90% LAS: 97.87%
dev:   Loss: 0.2458 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2543 UAS: 95.84% LAS: 94.18%
0:04:45.021867s elapsed

Epoch 219 / 1000:
train: Loss: 0.0605 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2452 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2519 UAS: 95.83% LAS: 94.14%
0:04:46.158972s elapsed

Epoch 220 / 1000:
train: Loss: 0.0605 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2448 UAS: 95.55% LAS: 93.73%
test:  Loss: 0.2512 UAS: 95.85% LAS: 94.17%
0:04:45.295402s elapsed

Epoch 221 / 1000:
train: Loss: 0.0604 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2449 UAS: 95.57% LAS: 93.75%
test:  Loss: 0.2517 UAS: 95.86% LAS: 94.19%
0:04:47.040927s elapsed

Epoch 222 / 1000:
train: Loss: 0.0603 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2464 UAS: 95.57% LAS: 93.75%
test:  Loss: 0.2535 UAS: 95.86% LAS: 94.18%
0:04:47.070568s elapsed

Epoch 223 / 1000:
train: Loss: 0.0601 UAS: 98.91% LAS: 97.89%
dev:   Loss: 0.2467 UAS: 95.58% LAS: 93.73%
test:  Loss: 0.2537 UAS: 95.87% LAS: 94.18%
0:04:46.172292s elapsed

Epoch 224 / 1000:
train: Loss: 0.0600 UAS: 98.92% LAS: 97.90%
dev:   Loss: 0.2457 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2529 UAS: 95.85% LAS: 94.17%
0:04:46.056736s elapsed

Epoch 225 / 1000:
train: Loss: 0.0601 UAS: 98.92% LAS: 97.90%
dev:   Loss: 0.2452 UAS: 95.57% LAS: 93.80%
test:  Loss: 0.2509 UAS: 95.87% LAS: 94.21%
0:04:46.012114s elapsed

Epoch 226 / 1000:
train: Loss: 0.0597 UAS: 98.92% LAS: 97.91%
dev:   Loss: 0.2466 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2517 UAS: 95.90% LAS: 94.23%
0:04:47.083395s elapsed

Epoch 227 / 1000:
train: Loss: 0.0596 UAS: 98.92% LAS: 97.91%
dev:   Loss: 0.2473 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2535 UAS: 95.88% LAS: 94.22%
0:04:45.460049s elapsed

Epoch 228 / 1000:
train: Loss: 0.0595 UAS: 98.92% LAS: 97.91%
dev:   Loss: 0.2482 UAS: 95.55% LAS: 93.74%
test:  Loss: 0.2537 UAS: 95.86% LAS: 94.19%
0:04:46.357340s elapsed

Epoch 229 / 1000:
train: Loss: 0.0593 UAS: 98.92% LAS: 97.91%
dev:   Loss: 0.2488 UAS: 95.57% LAS: 93.75%
test:  Loss: 0.2542 UAS: 95.87% LAS: 94.20%
0:04:47.113804s elapsed

Epoch 230 / 1000:
train: Loss: 0.0594 UAS: 98.93% LAS: 97.91%
dev:   Loss: 0.2472 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2522 UAS: 95.84% LAS: 94.17%
0:04:46.269521s elapsed

Epoch 231 / 1000:
train: Loss: 0.0592 UAS: 98.93% LAS: 97.92%
dev:   Loss: 0.2476 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2528 UAS: 95.85% LAS: 94.17%
0:04:46.593065s elapsed

Epoch 232 / 1000:
train: Loss: 0.0590 UAS: 98.93% LAS: 97.92%
dev:   Loss: 0.2487 UAS: 95.56% LAS: 93.74%
test:  Loss: 0.2538 UAS: 95.85% LAS: 94.18%
0:04:46.138817s elapsed

Epoch 233 / 1000:
train: Loss: 0.0590 UAS: 98.94% LAS: 97.93%
dev:   Loss: 0.2476 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2520 UAS: 95.82% LAS: 94.15%
0:04:46.344671s elapsed

Epoch 234 / 1000:
train: Loss: 0.0594 UAS: 98.94% LAS: 97.93%
dev:   Loss: 0.2464 UAS: 95.58% LAS: 93.76%
test:  Loss: 0.2514 UAS: 95.84% LAS: 94.17%
0:04:45.948536s elapsed

Epoch 235 / 1000:
train: Loss: 0.0591 UAS: 98.94% LAS: 97.93%
dev:   Loss: 0.2478 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2536 UAS: 95.86% LAS: 94.19%
0:04:47.031747s elapsed

Epoch 236 / 1000:
train: Loss: 0.0589 UAS: 98.94% LAS: 97.94%
dev:   Loss: 0.2494 UAS: 95.58% LAS: 93.76%
test:  Loss: 0.2545 UAS: 95.88% LAS: 94.20%
0:04:46.631199s elapsed

Epoch 237 / 1000:
train: Loss: 0.0593 UAS: 98.94% LAS: 97.93%
dev:   Loss: 0.2472 UAS: 95.58% LAS: 93.79%
test:  Loss: 0.2527 UAS: 95.87% LAS: 94.21%
0:04:46.466631s elapsed

Epoch 238 / 1000:
train: Loss: 0.0592 UAS: 98.94% LAS: 97.93%
dev:   Loss: 0.2459 UAS: 95.61% LAS: 93.80%
test:  Loss: 0.2528 UAS: 95.86% LAS: 94.19%
0:04:46.055831s elapsed

Epoch 239 / 1000:
train: Loss: 0.0590 UAS: 98.94% LAS: 97.94%
dev:   Loss: 0.2463 UAS: 95.61% LAS: 93.78%
test:  Loss: 0.2530 UAS: 95.85% LAS: 94.18%
0:04:46.269593s elapsed

Epoch 240 / 1000:
train: Loss: 0.0587 UAS: 98.95% LAS: 97.95%
dev:   Loss: 0.2474 UAS: 95.59% LAS: 93.77%
test:  Loss: 0.2530 UAS: 95.86% LAS: 94.18%
0:04:45.126945s elapsed

Epoch 241 / 1000:
train: Loss: 0.0586 UAS: 98.95% LAS: 97.95%
dev:   Loss: 0.2481 UAS: 95.64% LAS: 93.82%
test:  Loss: 0.2539 UAS: 95.87% LAS: 94.19%
0:04:46.005516s elapsed

Epoch 242 / 1000:
train: Loss: 0.0590 UAS: 98.94% LAS: 97.94%
dev:   Loss: 0.2465 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2530 UAS: 95.87% LAS: 94.20%
0:04:46.522282s elapsed

Epoch 243 / 1000:
train: Loss: 0.0585 UAS: 98.95% LAS: 97.95%
dev:   Loss: 0.2470 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2534 UAS: 95.87% LAS: 94.21%
0:04:45.324611s elapsed

Epoch 244 / 1000:
train: Loss: 0.0585 UAS: 98.95% LAS: 97.95%
dev:   Loss: 0.2475 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2539 UAS: 95.88% LAS: 94.21%
0:04:46.395769s elapsed

Epoch 245 / 1000:
train: Loss: 0.0581 UAS: 98.96% LAS: 97.96%
dev:   Loss: 0.2472 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2541 UAS: 95.86% LAS: 94.20%
0:04:46.875755s elapsed

Epoch 246 / 1000:
train: Loss: 0.0582 UAS: 98.96% LAS: 97.96%
dev:   Loss: 0.2468 UAS: 95.56% LAS: 93.73%
test:  Loss: 0.2530 UAS: 95.86% LAS: 94.19%
0:04:46.977232s elapsed

Epoch 247 / 1000:
train: Loss: 0.0581 UAS: 98.96% LAS: 97.96%
dev:   Loss: 0.2475 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2540 UAS: 95.89% LAS: 94.21%
0:04:46.075906s elapsed

Epoch 248 / 1000:
train: Loss: 0.0581 UAS: 98.96% LAS: 97.97%
dev:   Loss: 0.2484 UAS: 95.58% LAS: 93.74%
test:  Loss: 0.2545 UAS: 95.84% LAS: 94.18%
0:04:45.895635s elapsed

Epoch 249 / 1000:
train: Loss: 0.0579 UAS: 98.96% LAS: 97.97%
dev:   Loss: 0.2480 UAS: 95.58% LAS: 93.76%
test:  Loss: 0.2540 UAS: 95.90% LAS: 94.24%
0:04:47.809094s elapsed

Epoch 250 / 1000:
train: Loss: 0.0578 UAS: 98.96% LAS: 97.97%
dev:   Loss: 0.2479 UAS: 95.58% LAS: 93.75%
test:  Loss: 0.2540 UAS: 95.86% LAS: 94.20%
0:04:46.305396s elapsed

Epoch 251 / 1000:
train: Loss: 0.0579 UAS: 98.96% LAS: 97.97%
dev:   Loss: 0.2467 UAS: 95.58% LAS: 93.75%
test:  Loss: 0.2535 UAS: 95.88% LAS: 94.20%
0:04:47.084612s elapsed

Epoch 252 / 1000:
train: Loss: 0.0577 UAS: 98.97% LAS: 97.97%
dev:   Loss: 0.2478 UAS: 95.58% LAS: 93.75%
test:  Loss: 0.2540 UAS: 95.87% LAS: 94.18%
0:04:46.598850s elapsed

Epoch 253 / 1000:
train: Loss: 0.0580 UAS: 98.96% LAS: 97.97%
dev:   Loss: 0.2478 UAS: 95.54% LAS: 93.72%
test:  Loss: 0.2543 UAS: 95.86% LAS: 94.19%
0:04:43.693513s elapsed

Epoch 254 / 1000:
train: Loss: 0.0575 UAS: 98.97% LAS: 97.97%
dev:   Loss: 0.2486 UAS: 95.53% LAS: 93.71%
test:  Loss: 0.2556 UAS: 95.86% LAS: 94.18%
0:04:46.979442s elapsed

Epoch 255 / 1000:
train: Loss: 0.0576 UAS: 98.97% LAS: 97.98%
dev:   Loss: 0.2474 UAS: 95.55% LAS: 93.72%
test:  Loss: 0.2543 UAS: 95.87% LAS: 94.20%
0:04:45.178996s elapsed

Epoch 256 / 1000:
train: Loss: 0.0580 UAS: 98.97% LAS: 97.98%
dev:   Loss: 0.2466 UAS: 95.55% LAS: 93.74%
test:  Loss: 0.2537 UAS: 95.84% LAS: 94.19%
0:04:46.235161s elapsed

Epoch 257 / 1000:
train: Loss: 0.0577 UAS: 98.97% LAS: 97.98%
dev:   Loss: 0.2471 UAS: 95.55% LAS: 93.73%
test:  Loss: 0.2532 UAS: 95.88% LAS: 94.21%
0:04:47.374313s elapsed

Epoch 258 / 1000:
train: Loss: 0.0577 UAS: 98.97% LAS: 97.98%
dev:   Loss: 0.2471 UAS: 95.57% LAS: 93.75%
test:  Loss: 0.2528 UAS: 95.85% LAS: 94.20%
0:04:47.042554s elapsed

Epoch 259 / 1000:
train: Loss: 0.0577 UAS: 98.97% LAS: 97.98%
dev:   Loss: 0.2473 UAS: 95.54% LAS: 93.71%
test:  Loss: 0.2533 UAS: 95.85% LAS: 94.19%
0:04:45.790878s elapsed

Epoch 260 / 1000:
train: Loss: 0.0575 UAS: 98.98% LAS: 97.99%
dev:   Loss: 0.2477 UAS: 95.55% LAS: 93.73%
test:  Loss: 0.2538 UAS: 95.86% LAS: 94.20%
0:04:47.229726s elapsed

Epoch 261 / 1000:
train: Loss: 0.0574 UAS: 98.98% LAS: 97.99%
dev:   Loss: 0.2477 UAS: 95.59% LAS: 93.76%
test:  Loss: 0.2542 UAS: 95.85% LAS: 94.19%
0:04:46.875788s elapsed

Epoch 262 / 1000:
train: Loss: 0.0572 UAS: 98.98% LAS: 97.99%
dev:   Loss: 0.2479 UAS: 95.57% LAS: 93.75%
test:  Loss: 0.2547 UAS: 95.86% LAS: 94.21%
0:04:46.187300s elapsed

Epoch 263 / 1000:
train: Loss: 0.0571 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2487 UAS: 95.58% LAS: 93.73%
test:  Loss: 0.2554 UAS: 95.87% LAS: 94.22%
0:04:47.059984s elapsed

Epoch 264 / 1000:
train: Loss: 0.0573 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2469 UAS: 95.57% LAS: 93.73%
test:  Loss: 0.2541 UAS: 95.88% LAS: 94.23%
0:04:45.871466s elapsed

Epoch 265 / 1000:
train: Loss: 0.0570 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2479 UAS: 95.55% LAS: 93.73%
test:  Loss: 0.2550 UAS: 95.89% LAS: 94.24%
0:04:46.925861s elapsed

Epoch 266 / 1000:
train: Loss: 0.0572 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2470 UAS: 95.57% LAS: 93.74%
test:  Loss: 0.2539 UAS: 95.88% LAS: 94.22%
0:04:46.926306s elapsed

Epoch 267 / 1000:
train: Loss: 0.0570 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2470 UAS: 95.55% LAS: 93.72%
test:  Loss: 0.2552 UAS: 95.87% LAS: 94.21%
0:04:48.205795s elapsed

Epoch 268 / 1000:
train: Loss: 0.0569 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2479 UAS: 95.55% LAS: 93.73%
test:  Loss: 0.2553 UAS: 95.87% LAS: 94.20%
0:04:46.651287s elapsed

Epoch 269 / 1000:
train: Loss: 0.0570 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2474 UAS: 95.55% LAS: 93.74%
test:  Loss: 0.2542 UAS: 95.88% LAS: 94.22%
0:04:46.938530s elapsed

Epoch 270 / 1000:
train: Loss: 0.0568 UAS: 98.99% LAS: 98.00%
dev:   Loss: 0.2483 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2543 UAS: 95.86% LAS: 94.20%
0:04:46.444310s elapsed

Epoch 271 / 1000:
train: Loss: 0.0568 UAS: 98.98% LAS: 98.00%
dev:   Loss: 0.2481 UAS: 95.56% LAS: 93.74%
test:  Loss: 0.2544 UAS: 95.85% LAS: 94.19%
0:04:47.394997s elapsed

Epoch 272 / 1000:
train: Loss: 0.0568 UAS: 98.99% LAS: 98.01%
dev:   Loss: 0.2477 UAS: 95.60% LAS: 93.78%
test:  Loss: 0.2549 UAS: 95.85% LAS: 94.20%
0:04:46.065496s elapsed

Epoch 273 / 1000:
train: Loss: 0.0569 UAS: 98.99% LAS: 98.01%
dev:   Loss: 0.2476 UAS: 95.60% LAS: 93.80%
test:  Loss: 0.2545 UAS: 95.86% LAS: 94.20%
0:04:46.386472s elapsed

Epoch 274 / 1000:
train: Loss: 0.0566 UAS: 98.99% LAS: 98.01%
dev:   Loss: 0.2481 UAS: 95.59% LAS: 93.79%
test:  Loss: 0.2550 UAS: 95.85% LAS: 94.19%
0:04:44.800246s elapsed

Epoch 275 / 1000:
train: Loss: 0.0570 UAS: 98.99% LAS: 98.01%
dev:   Loss: 0.2480 UAS: 95.54% LAS: 93.73%
test:  Loss: 0.2551 UAS: 95.86% LAS: 94.21%
0:04:46.779237s elapsed

Epoch 276 / 1000:
train: Loss: 0.0568 UAS: 98.99% LAS: 98.01%
dev:   Loss: 0.2478 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2554 UAS: 95.85% LAS: 94.19%
0:04:45.798942s elapsed

Epoch 277 / 1000:
train: Loss: 0.0567 UAS: 99.00% LAS: 98.02%
dev:   Loss: 0.2486 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2558 UAS: 95.88% LAS: 94.21%
0:04:45.869679s elapsed

Epoch 278 / 1000:
train: Loss: 0.0565 UAS: 99.00% LAS: 98.01%
dev:   Loss: 0.2483 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2554 UAS: 95.86% LAS: 94.20%
0:04:46.151278s elapsed

Epoch 279 / 1000:
train: Loss: 0.0563 UAS: 99.00% LAS: 98.02%
dev:   Loss: 0.2480 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2556 UAS: 95.87% LAS: 94.21%
0:04:46.271111s elapsed

Epoch 280 / 1000:
train: Loss: 0.0565 UAS: 99.00% LAS: 98.02%
dev:   Loss: 0.2473 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2547 UAS: 95.87% LAS: 94.19%
0:04:47.425761s elapsed

Epoch 281 / 1000:
train: Loss: 0.0565 UAS: 98.99% LAS: 98.02%
dev:   Loss: 0.2471 UAS: 95.56% LAS: 93.77%
test:  Loss: 0.2550 UAS: 95.89% LAS: 94.21%
0:04:46.394572s elapsed

Epoch 282 / 1000:
train: Loss: 0.0562 UAS: 99.00% LAS: 98.02%
dev:   Loss: 0.2480 UAS: 95.57% LAS: 93.78%
test:  Loss: 0.2554 UAS: 95.87% LAS: 94.20%
0:04:45.835981s elapsed

Epoch 283 / 1000:
train: Loss: 0.0563 UAS: 99.00% LAS: 98.02%
dev:   Loss: 0.2491 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2558 UAS: 95.89% LAS: 94.21%
0:04:47.036759s elapsed

Epoch 284 / 1000:
train: Loss: 0.0564 UAS: 99.00% LAS: 98.03%
dev:   Loss: 0.2478 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2542 UAS: 95.88% LAS: 94.21%
0:04:46.642373s elapsed

Epoch 285 / 1000:
train: Loss: 0.0562 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2479 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2540 UAS: 95.87% LAS: 94.21%
0:04:46.796901s elapsed

Epoch 286 / 1000:
train: Loss: 0.0560 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2486 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2545 UAS: 95.87% LAS: 94.21%
0:04:46.822122s elapsed

Epoch 287 / 1000:
train: Loss: 0.0560 UAS: 99.01% LAS: 98.03%
dev:   Loss: 0.2489 UAS: 95.54% LAS: 93.75%
test:  Loss: 0.2544 UAS: 95.86% LAS: 94.19%
0:04:45.857528s elapsed

Epoch 288 / 1000:
train: Loss: 0.0558 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2485 UAS: 95.57% LAS: 93.78%
test:  Loss: 0.2542 UAS: 95.88% LAS: 94.21%
0:04:46.408475s elapsed

Epoch 289 / 1000:
train: Loss: 0.0559 UAS: 99.00% LAS: 98.04%
dev:   Loss: 0.2490 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2546 UAS: 95.87% LAS: 94.21%
0:04:45.452681s elapsed

Epoch 290 / 1000:
train: Loss: 0.0562 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2484 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2532 UAS: 95.88% LAS: 94.21%
0:04:46.726287s elapsed

Epoch 291 / 1000:
train: Loss: 0.0560 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2489 UAS: 95.58% LAS: 93.75%
test:  Loss: 0.2539 UAS: 95.88% LAS: 94.21%
0:04:46.704770s elapsed

Epoch 292 / 1000:
train: Loss: 0.0556 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2502 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2553 UAS: 95.88% LAS: 94.21%
0:04:46.322354s elapsed

Epoch 293 / 1000:
train: Loss: 0.0556 UAS: 99.01% LAS: 98.05%
dev:   Loss: 0.2492 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2548 UAS: 95.88% LAS: 94.22%
0:04:46.915090s elapsed

Epoch 294 / 1000:
train: Loss: 0.0556 UAS: 99.01% LAS: 98.05%
dev:   Loss: 0.2491 UAS: 95.58% LAS: 93.75%
test:  Loss: 0.2547 UAS: 95.88% LAS: 94.21%
0:04:47.988831s elapsed

Epoch 295 / 1000:
train: Loss: 0.0556 UAS: 99.01% LAS: 98.05%
dev:   Loss: 0.2496 UAS: 95.56% LAS: 93.74%
test:  Loss: 0.2559 UAS: 95.87% LAS: 94.20%
0:04:45.489542s elapsed

Epoch 296 / 1000:
train: Loss: 0.0558 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2484 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2549 UAS: 95.86% LAS: 94.20%
0:04:46.305496s elapsed

Epoch 297 / 1000:
train: Loss: 0.0558 UAS: 99.01% LAS: 98.04%
dev:   Loss: 0.2491 UAS: 95.56% LAS: 93.74%
test:  Loss: 0.2548 UAS: 95.86% LAS: 94.20%
0:04:47.495048s elapsed

Epoch 298 / 1000:
train: Loss: 0.0556 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2491 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2552 UAS: 95.86% LAS: 94.21%
0:04:45.134158s elapsed

Epoch 299 / 1000:
train: Loss: 0.0556 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2485 UAS: 95.57% LAS: 93.75%
test:  Loss: 0.2544 UAS: 95.88% LAS: 94.23%
0:04:46.279020s elapsed

Epoch 300 / 1000:
train: Loss: 0.0555 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2488 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2548 UAS: 95.85% LAS: 94.19%
0:04:45.976943s elapsed

Epoch 301 / 1000:
train: Loss: 0.0555 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2489 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2546 UAS: 95.86% LAS: 94.21%
0:04:46.623007s elapsed

Epoch 302 / 1000:
train: Loss: 0.0556 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2485 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2546 UAS: 95.85% LAS: 94.19%
0:04:47.813401s elapsed

Epoch 303 / 1000:
train: Loss: 0.0554 UAS: 99.02% LAS: 98.06%
dev:   Loss: 0.2490 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2552 UAS: 95.86% LAS: 94.20%
0:04:46.084385s elapsed

Epoch 304 / 1000:
train: Loss: 0.0555 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2490 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2553 UAS: 95.85% LAS: 94.18%
0:04:45.559187s elapsed

Epoch 305 / 1000:
train: Loss: 0.0554 UAS: 99.02% LAS: 98.05%
dev:   Loss: 0.2494 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2559 UAS: 95.88% LAS: 94.22%
0:04:46.184101s elapsed

Epoch 306 / 1000:
train: Loss: 0.0554 UAS: 99.02% LAS: 98.06%
dev:   Loss: 0.2491 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2548 UAS: 95.87% LAS: 94.21%
0:04:46.463838s elapsed

Epoch 307 / 1000:
train: Loss: 0.0554 UAS: 99.03% LAS: 98.06%
dev:   Loss: 0.2488 UAS: 95.55% LAS: 93.74%
test:  Loss: 0.2550 UAS: 95.88% LAS: 94.22%
0:04:46.414567s elapsed

Epoch 308 / 1000:
train: Loss: 0.0553 UAS: 99.03% LAS: 98.06%
dev:   Loss: 0.2495 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2558 UAS: 95.86% LAS: 94.20%
0:04:47.079337s elapsed

Epoch 309 / 1000:
train: Loss: 0.0553 UAS: 99.03% LAS: 98.06%
dev:   Loss: 0.2487 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2548 UAS: 95.86% LAS: 94.20%
0:04:47.339364s elapsed

Epoch 310 / 1000:
train: Loss: 0.0552 UAS: 99.02% LAS: 98.06%
dev:   Loss: 0.2494 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2556 UAS: 95.87% LAS: 94.22%
0:04:46.315009s elapsed

Epoch 311 / 1000:
train: Loss: 0.0553 UAS: 99.03% LAS: 98.06%
dev:   Loss: 0.2493 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2553 UAS: 95.87% LAS: 94.20%
0:04:47.427864s elapsed

Epoch 312 / 1000:
train: Loss: 0.0552 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2486 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2550 UAS: 95.87% LAS: 94.20%
0:04:47.108074s elapsed

Epoch 313 / 1000:
train: Loss: 0.0551 UAS: 99.03% LAS: 98.06%
dev:   Loss: 0.2492 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2559 UAS: 95.85% LAS: 94.19%
0:04:47.673461s elapsed

Epoch 314 / 1000:
train: Loss: 0.0549 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2494 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2558 UAS: 95.87% LAS: 94.21%
0:04:47.132105s elapsed

Epoch 315 / 1000:
train: Loss: 0.0549 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2498 UAS: 95.57% LAS: 93.74%
test:  Loss: 0.2561 UAS: 95.87% LAS: 94.21%
0:04:45.695687s elapsed

Epoch 316 / 1000:
train: Loss: 0.0550 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2496 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2559 UAS: 95.86% LAS: 94.20%
0:04:46.449944s elapsed

Epoch 317 / 1000:
train: Loss: 0.0551 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2497 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2558 UAS: 95.85% LAS: 94.19%
0:04:45.815201s elapsed

Epoch 318 / 1000:
train: Loss: 0.0550 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2492 UAS: 95.56% LAS: 93.75%
test:  Loss: 0.2551 UAS: 95.84% LAS: 94.20%
0:04:46.113135s elapsed

Epoch 319 / 1000:
train: Loss: 0.0549 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2496 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2557 UAS: 95.85% LAS: 94.19%
0:04:47.366536s elapsed

Epoch 320 / 1000:
train: Loss: 0.0548 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2495 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2557 UAS: 95.86% LAS: 94.21%
0:04:45.858419s elapsed

Epoch 321 / 1000:
train: Loss: 0.0549 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2492 UAS: 95.59% LAS: 93.78%
test:  Loss: 0.2551 UAS: 95.86% LAS: 94.21%
0:04:45.158248s elapsed

Epoch 322 / 1000:
train: Loss: 0.0549 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2492 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2550 UAS: 95.87% LAS: 94.23%
0:04:45.579440s elapsed

Epoch 323 / 1000:
train: Loss: 0.0548 UAS: 99.03% LAS: 98.07%
dev:   Loss: 0.2503 UAS: 95.59% LAS: 93.79%
test:  Loss: 0.2563 UAS: 95.88% LAS: 94.24%
0:04:45.508492s elapsed

Epoch 324 / 1000:
train: Loss: 0.0547 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2494 UAS: 95.58% LAS: 93.78%
test:  Loss: 0.2556 UAS: 95.86% LAS: 94.22%
0:04:46.243830s elapsed

Epoch 325 / 1000:
train: Loss: 0.0546 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2493 UAS: 95.56% LAS: 93.76%
test:  Loss: 0.2556 UAS: 95.86% LAS: 94.23%
0:04:45.379917s elapsed

Epoch 326 / 1000:
train: Loss: 0.0547 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2490 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2551 UAS: 95.87% LAS: 94.23%
0:04:46.360100s elapsed

Epoch 327 / 1000:
train: Loss: 0.0546 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2491 UAS: 95.58% LAS: 93.79%
test:  Loss: 0.2551 UAS: 95.88% LAS: 94.24%
0:04:46.162675s elapsed

Epoch 328 / 1000:
train: Loss: 0.0546 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2495 UAS: 95.58% LAS: 93.78%
test:  Loss: 0.2558 UAS: 95.87% LAS: 94.23%
0:04:45.361226s elapsed

Epoch 329 / 1000:
train: Loss: 0.0547 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2493 UAS: 95.59% LAS: 93.80%
test:  Loss: 0.2555 UAS: 95.89% LAS: 94.24%
0:04:45.847511s elapsed

Epoch 330 / 1000:
train: Loss: 0.0546 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2491 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2556 UAS: 95.89% LAS: 94.24%
0:04:46.137228s elapsed

Epoch 331 / 1000:
train: Loss: 0.0546 UAS: 99.04% LAS: 98.09%
dev:   Loss: 0.2496 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2554 UAS: 95.87% LAS: 94.22%
0:04:46.234806s elapsed

Epoch 332 / 1000:
train: Loss: 0.0546 UAS: 99.04% LAS: 98.08%
dev:   Loss: 0.2489 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2553 UAS: 95.87% LAS: 94.23%
0:04:46.366033s elapsed

Epoch 333 / 1000:
train: Loss: 0.0546 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2487 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2548 UAS: 95.86% LAS: 94.21%
0:04:45.768164s elapsed

Epoch 334 / 1000:
train: Loss: 0.0546 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2491 UAS: 95.58% LAS: 93.77%
test:  Loss: 0.2554 UAS: 95.87% LAS: 94.23%
0:04:46.425745s elapsed

Epoch 335 / 1000:
train: Loss: 0.0546 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2496 UAS: 95.57% LAS: 93.78%
test:  Loss: 0.2558 UAS: 95.86% LAS: 94.22%
0:04:46.789604s elapsed

Epoch 336 / 1000:
train: Loss: 0.0545 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2490 UAS: 95.55% LAS: 93.75%
test:  Loss: 0.2552 UAS: 95.85% LAS: 94.21%
0:04:46.652024s elapsed

Epoch 337 / 1000:
train: Loss: 0.0545 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2493 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2560 UAS: 95.85% LAS: 94.21%
0:04:47.572142s elapsed

Epoch 338 / 1000:
train: Loss: 0.0545 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2490 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2554 UAS: 95.86% LAS: 94.22%
0:04:46.614200s elapsed

Epoch 339 / 1000:
train: Loss: 0.0545 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2483 UAS: 95.57% LAS: 93.77%
test:  Loss: 0.2551 UAS: 95.86% LAS: 94.22%
0:04:45.972352s elapsed

Epoch 340 / 1000:
train: Loss: 0.0546 UAS: 99.05% LAS: 98.10%
dev:   Loss: 0.2484 UAS: 95.58% LAS: 93.78%
test:  Loss: 0.2551 UAS: 95.86% LAS: 94.20%
0:04:47.337308s elapsed

Epoch 341 / 1000:
train: Loss: 0.0543 UAS: 99.05% LAS: 98.09%
dev:   Loss: 0.2501 UAS: 95.57% LAS: 93.76%
test:  Loss: 0.2568 UAS: 95.87% LAS: 94.21%
0:04:46.410133s elapsed

max score of dev is 93.82% at epoch 241
the score of test at epoch 241 is 94.19%
mean time of each epoch is 0:04:46.561768s
1 day, 3:08:37.562873s elapsed
